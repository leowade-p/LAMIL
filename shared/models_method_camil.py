import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ComponentViTEncoder

try:
    from nystrom_attention import NystromAttention
except ImportError as exc:
    raise ImportError(
        "CAMIL requires `nystrom-attention`. Install with: pip install nystrom-attention"
    ) from exc


def _initialize_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class _TransLayer(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class _CAM(nn.Module):
    def __init__(self, n_channel: int, mlp_r: int = 2, temperature: float = 1.0):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(n_channel, n_channel // mlp_r),
            nn.ReLU(),
            nn.Linear(n_channel // mlp_r, 3 * n_channel),
        )
        self.temperature = temperature

    def forward(self, x: torch.Tensor):
        bsz, channels, _, _ = x.shape
        max_feat = self.max_pool(x)
        avg_feat = self.avg_pool(x)
        max_feat = self.linear(max_feat.view(bsz, channels)).view(bsz, 3 * channels, 1, 1)
        avg_feat = self.linear(avg_feat.view(bsz, channels)).view(bsz, 3 * channels, 1, 1)
        y = torch.sigmoid((max_feat + avg_feat) * self.temperature)
        return y[:, :channels], y[:, channels : 2 * channels], y[:, 2 * channels :]


class _MCAB(nn.Module):
    def __init__(self, dim: int = 512, temperature: float = 3.0):
        super().__init__()
        self.local_feature3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=1),
        )
        self.local_feature5 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=1),
        )
        self.cam = _CAM(n_channel=dim, mlp_r=2, temperature=temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, channels = x.shape
        side_h = int(np.ceil(np.sqrt(num_tokens)))
        side_w = side_h
        pad_len = side_h * side_w - num_tokens
        if pad_len > 0:
            zero_tensor = torch.zeros(bsz, pad_len, channels, device=x.device, dtype=x.dtype)
            x_pad = torch.cat([x, zero_tensor], dim=1)
        else:
            x_pad = x

        cnn_feat = x_pad.transpose(1, 2).view(bsz, channels, side_h, side_w)
        x3_3 = self.local_feature3(cnn_feat)
        x5_5 = self.local_feature5(cnn_feat)
        fused = cnn_feat + x3_3 + x5_5

        y1, y2, y3 = self.cam(fused)
        out = y1 * cnn_feat + y2 * x3_3 + y3 * x5_5
        out = out.flatten(2).transpose(1, 2)
        return out[:, :num_tokens]


class _CAMILBlock(nn.Module):
    def __init__(self, dim: int = 512, temperature: float = 3.0):
        super().__init__()
        self.trans_layer = _TransLayer(dim=dim)
        self.mcab = _MCAB(dim=dim, temperature=temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trans_layer(x)
        return self.mcab(x)


class _AttnNet(nn.Module):
    def __init__(self, in_dim: int = 512, attn_dim: int = 256, dropout: bool = False):
        super().__init__()
        layers = [nn.Linear(in_dim, attn_dim), nn.Tanh()]
        if dropout:
            layers.append(nn.Dropout(0.25))
        layers.append(nn.Linear(attn_dim, 1))
        self.module = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.module(x), x


class _AttnNetGated(nn.Module):
    def __init__(self, in_dim: int = 512, attn_dim: int = 256, dropout: bool = False):
        super().__init__()
        attn_a = [nn.Linear(in_dim, attn_dim), nn.Tanh()]
        attn_b = [nn.Linear(in_dim, attn_dim), nn.Sigmoid()]
        if dropout:
            attn_a.append(nn.Dropout(0.25))
            attn_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*attn_a)
        self.attention_b = nn.Sequential(*attn_b)
        self.attention_c = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor):
        a = self.attention_a(x)
        b = self.attention_b(x)
        attn = self.attention_c(a * b)
        return attn, x


class _CAMILCore(nn.Module):
    """
    CAMIL aggregation core adapted from the official implementation.
    Expects one patient bag per forward call: (1, N, input_dim).
    """

    def __init__(
        self,
        n_classes: int = 2,
        input_dim: int = 128,
        agg_dim: int = 512,
        temperature: float = 1.2,
        dropout: float = 0.15,
        n_layers: int = 4,
        gate: bool = True,
        attention_dim: int = 256,
    ):
        super().__init__()
        self.n_classes = n_classes
        self._fc1 = nn.Sequential(
            nn.Linear(input_dim, agg_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [_CAMILBlock(dim=agg_dim, temperature=temperature) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(agg_dim)
        self.attention_net = (
            _AttnNetGated(in_dim=agg_dim, attn_dim=attention_dim, dropout=False)
            if gate
            else _AttnNet(in_dim=agg_dim, attn_dim=attention_dim, dropout=False)
        )
        self._fc2 = nn.Linear(agg_dim, n_classes)
        _initialize_weights(self)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (1, N, input_dim)
        h = self._fc1(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h).squeeze(0)

        attn, h = self.attention_net(h)
        attn = attn.permute(1, 0)
        attn = F.softmax(attn, dim=1)
        bag_feat = torch.mm(attn, h)
        return self._fc2(bag_feat).squeeze(0)


class CAMILClassifier(nn.Module):
    """
    CAMIL adapted to the current patient-component batch format.

    Input batch keys expected:
      - sequences: (N_components, MAX_ROIS, input_dim)
      - masks: (N_components, MAX_ROIS)
      - component_patient_indices: (N_components,)
      - num_patients: int
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_classes: int = 2,
        agg_dim: int = 512,
        camil_n_layers: int = 4,
        temperature: float = 1.2,
        dropout: float = 0.15,
        gate: bool = True,
        attention_dim: int = 256,
    ):
        super().__init__()
        self.component_encoder = ComponentViTEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        self.component_classifier = nn.Linear(hidden_dim, n_classes)
        self.instance_classifier = nn.Linear(hidden_dim, n_classes)
        self.camil_core = _CAMILCore(
            n_classes=n_classes,
            input_dim=hidden_dim,
            agg_dim=agg_dim,
            temperature=temperature,
            dropout=dropout,
            n_layers=camil_n_layers,
            gate=gate,
            attention_dim=attention_dim,
        )

    def _patient_forward(self, patient_tokens: torch.Tensor) -> torch.Tensor:
        return self.camil_core(patient_tokens.unsqueeze(0))

    def forward(self, batch: dict):
        sequences = batch["sequences"]
        masks = batch["masks"]
        component_patient_indices = batch["component_patient_indices"]
        num_patients = int(batch["num_patients"])

        encoded = self.component_encoder(sequences, masks)
        cls_outputs = encoded["cls_output"]
        roi_outputs = encoded["roi_outputs"]

        bag_logits = self.component_classifier(cls_outputs)

        patient_logits = []
        for p_idx in range(num_patients):
            p_mask = component_patient_indices == p_idx
            if torch.any(p_mask):
                patient_logits.append(self._patient_forward(cls_outputs[p_mask]))
            else:
                patient_logits.append(torch.zeros(bag_logits.shape[1], device=cls_outputs.device))
        patient_logits = torch.stack(patient_logits, dim=0)

        bsz, seq_len, _ = roi_outputs.shape
        if masks.shape[1] != seq_len:
            if masks.shape[1] > seq_len:
                masks = masks[:, :seq_len]
            else:
                padding = torch.zeros(
                    bsz,
                    seq_len - masks.shape[1],
                    dtype=torch.bool,
                    device=masks.device,
                )
                masks = torch.cat([masks, padding], dim=1)

        valid_roi_outputs = roi_outputs[masks]
        instance_logits = self.instance_classifier(valid_roi_outputs)

        return {
            "bag_logits": bag_logits,
            "instance_logits": instance_logits,
            "patient_logits": patient_logits,
        }
