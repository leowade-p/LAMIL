import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ComponentViTEncoder


class _PPEG(nn.Module):
    """
    Pyramid Position Encoding Generator (lightweight variant).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj_3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.proj_5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.proj_7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, N+1, D), token0 is cls
        cls_tok, patch_tok = x[:, :1, :], x[:, 1:, :]
        bsz, num_patch, dim = patch_tok.shape
        side = int(math.sqrt(num_patch))

        patch_2d = patch_tok.transpose(1, 2).reshape(bsz, dim, side, side)
        fused = patch_2d + self.proj_3(patch_2d) + self.proj_5(patch_2d) + self.proj_7(patch_2d)
        fused_tok = fused.reshape(bsz, dim, num_patch).transpose(1, 2)
        return torch.cat([cls_tok, fused_tok], dim=1)


class TransMILClassifier(nn.Module):
    """
    TransMIL-style correlated MIL aggregator (practical adaptation).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_classes: int = 2,
        ff_mult: int = 4,
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

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.block1 = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * ff_mult,
            batch_first=True,
        )
        self.ppeg = _PPEG(hidden_dim)
        self.block2 = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * ff_mult,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.patient_head = nn.Linear(hidden_dim, n_classes)

    @staticmethod
    def _square_pad_tokens(tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (K, D)
        k, _ = tokens.shape
        side = int(math.ceil(math.sqrt(k)))
        n = side * side
        m = n - k
        if m <= 0:
            return tokens
        # Paper uses duplication for square padding.
        dup = tokens[:m]
        return torch.cat([tokens, dup], dim=0)

    def _patient_forward(self, patient_tokens: torch.Tensor):
        # patient_tokens: (K, D)
        patient_tokens = self._square_pad_tokens(patient_tokens)  # (N, D), N perfect square
        x = patient_tokens.unsqueeze(0)  # (1, N, D)
        cls_tok = self.cls_token.expand(1, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)  # (1, N+1, D)

        x = self.block1(x)
        x = self.ppeg(x)
        x = self.block2(x)
        x = self.norm(x)

        cls_out = x[:, 0, :]  # (1, D)
        logits = self.patient_head(cls_out).squeeze(0)  # (C,)
        return logits

    def forward(self, batch: dict):
        sequences = batch["sequences"]
        masks = batch["masks"]
        component_patient_indices = batch["component_patient_indices"]
        num_patients = int(batch["num_patients"])

        encoded = self.component_encoder(sequences, masks)
        cls_outputs = encoded["cls_output"]   # (N_components, hidden)
        roi_outputs = encoded["roi_outputs"]  # (N_components, MAX_ROIS, hidden)

        bag_logits = self.component_classifier(cls_outputs)

        # Patient-level TransMIL aggregation
        patient_logits = []
        for p_idx in range(num_patients):
            mask = component_patient_indices == p_idx
            if torch.any(mask):
                patient_logits.append(self._patient_forward(cls_outputs[mask]))
            else:
                patient_logits.append(torch.zeros(bag_logits.shape[1], device=cls_outputs.device))
        patient_logits = torch.stack(patient_logits, dim=0)

        # ROI-level compatibility output
        bsz, seq_len, _ = roi_outputs.shape
        if masks.shape[1] != seq_len:
            if masks.shape[1] > seq_len:
                masks = masks[:, :seq_len]
            else:
                pad = torch.zeros(
                    bsz, seq_len - masks.shape[1], dtype=torch.bool, device=masks.device
                )
                masks = torch.cat([masks, pad], dim=1)
        valid_roi_outputs = roi_outputs[masks]
        instance_logits = self.instance_classifier(valid_roi_outputs)

        return {
            "bag_logits": bag_logits,
            "instance_logits": instance_logits,
            "patient_logits": patient_logits,
        }

