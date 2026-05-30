import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ComponentViTEncoder


class ABMILClassifier(nn.Module):
    """
    Attention-based Deep MIL (ABMIL) adapted to current project batches.

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
        attention_dim: int = 128,
        gated: bool = True,
    ):
        super().__init__()
        self.component_encoder = ComponentViTEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        self.gated = gated

        # Optional component-level head (kept for compatibility with existing losses).
        self.component_classifier = nn.Linear(hidden_dim, n_classes)

        # ABMIL attention network.
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        if gated:
            self.attention_u = nn.Linear(hidden_dim, attention_dim)
        else:
            self.attention_u = None

        # Patient/bag-level classifier.
        self.bag_classifier = nn.Linear(hidden_dim, n_classes)

        # Optional ROI-level head (kept for compatibility with existing auxiliary loss).
        self.instance_classifier = nn.Linear(hidden_dim, n_classes)

    def _patient_attention_pool(
        self,
        component_embeddings: torch.Tensor,
        component_patient_indices: torch.Tensor,
        num_patients: int,
    ):
        """
        Per-patient softmax attention pooling.
        """
        device = component_embeddings.device
        n_components, hidden_dim = component_embeddings.shape
        attention_weights = torch.zeros(n_components, device=device)
        bag_embeddings = torch.zeros(num_patients, hidden_dim, device=device)

        for patient_idx in range(num_patients):
            mask = component_patient_indices == patient_idx
            if not torch.any(mask):
                continue

            h_i = component_embeddings[mask]  # (Ki, D)
            v = torch.tanh(self.attention_v(h_i))
            if self.gated:
                u = torch.sigmoid(self.attention_u(h_i))
                attn_logits = self.attention_w(v * u).squeeze(-1)
            else:
                attn_logits = self.attention_w(v).squeeze(-1)

            a_i = F.softmax(attn_logits, dim=0)
            z_i = torch.sum(a_i.unsqueeze(-1) * h_i, dim=0)

            attention_weights[mask] = a_i
            bag_embeddings[patient_idx] = z_i

        return bag_embeddings, attention_weights

    def forward(self, batch: dict):
        sequences = batch["sequences"]
        masks = batch["masks"]
        component_patient_indices = batch["component_patient_indices"]
        num_patients = int(batch["num_patients"])

        encoded = self.component_encoder(sequences, masks)
        cls_outputs = encoded["cls_output"]   # (N_components, hidden_dim)
        roi_outputs = encoded["roi_outputs"]  # (N_components, MAX_ROIS, hidden_dim)

        # Component-level logits
        bag_logits = self.component_classifier(cls_outputs)

        # ABMIL patient-level logits
        bag_embeddings, attention_weights = self._patient_attention_pool(
            component_embeddings=cls_outputs,
            component_patient_indices=component_patient_indices,
            num_patients=num_patients,
        )
        patient_logits = self.bag_classifier(bag_embeddings)

        # ROI-level logits for optional auxiliary supervision
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
            "bag_logits": bag_logits,                # per-component logits
            "instance_logits": instance_logits,      # per-ROI logits
            "patient_logits": patient_logits,        # per-patient logits
            "attention_weights": attention_weights,  # per-component attention
        }

