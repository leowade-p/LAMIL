import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ComponentViTEncoder


class DSMILClassifier(nn.Module):
    """
    Dual-stream MIL classifier adapted from:
    "Dual-stream Multiple Instance Learning Network (DSMIL)".

    This implementation is aligned to the official dsmil.py logic:
      - Stream-1: per-instance/component classifier and per-class max instance logits.
      - Stream-2: BClassifier-style attention pooling with per-class critical instances.
      - Train objective can use 0.5 * CE(max_logits) + 0.5 * CE(bag_logits).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_classes: int = 2,
        q_dim: int = 128,
        nonlinear: bool = True,
        passing_v: bool = False,
        dropout_v: float = 0.0,
    ):
        super().__init__()
        self.component_encoder = ComponentViTEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.q_dim = q_dim

        # Stream-1: instance/component classifier (IClassifier equivalent)
        self.component_classifier = nn.Linear(hidden_dim, n_classes)

        # Stream-2: BClassifier equivalent
        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(hidden_dim, q_dim),
                nn.ReLU(),
                nn.Linear(q_dim, q_dim),
                nn.Tanh(),
            )
        else:
            self.q = nn.Linear(hidden_dim, q_dim)

        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            self.v = nn.Identity()
        self.fcc = nn.Conv1d(n_classes, n_classes, kernel_size=hidden_dim)

        # Optional ROI-level head for compatibility with existing aux losses
        self.instance_classifier = nn.Linear(hidden_dim, n_classes)

    def _dsmil_pool_per_patient(
        self,
        component_embeddings: torch.Tensor,
        component_logits: torch.Tensor,
        component_patient_indices: torch.Tensor,
        num_patients: int,
    ):
        device = component_embeddings.device
        n_components, _ = component_embeddings.shape
        scale = torch.sqrt(torch.tensor(float(self.q_dim), device=device))

        # Bag branch logits per patient (C)
        patient_bag_logits = torch.zeros(num_patients, self.n_classes, device=device)
        # Max instance logits per patient (C)
        patient_max_logits = torch.zeros(num_patients, self.n_classes, device=device)

        # For visualization/debug only: keep per-component attention for positive class
        attention_weights = torch.zeros(n_components, device=device)
        critical_component_indices = torch.full(
            (num_patients,), -1, dtype=torch.long, device=device
        )

        for p_idx in range(num_patients):
            mask = component_patient_indices == p_idx
            if not torch.any(mask):
                continue

            feats_i = component_embeddings[mask]      # (K, D)
            classes_i = component_logits[mask]        # (K, C)

            # Stream-1: max over instances for each class
            max_logits_i, _ = torch.max(classes_i, dim=0)  # (C,)
            patient_max_logits[p_idx] = max_logits_i

            # BClassifier logic from official DSMIL
            V = self.v(feats_i)                 # (K, D)
            Q = self.q(feats_i).view(feats_i.shape[0], -1)  # (K, 128)

            # Critical instance per class by sorting class logits along instances
            _, m_indices = torch.sort(classes_i, 0, descending=True)  # (K, C)
            m_feats = torch.index_select(feats_i, dim=0, index=m_indices[0, :])  # (C, D)
            q_max = self.q(m_feats)  # (C, 128)

            # Attention matrix A: (K, C), softmax over instances (dim=0)
            A = torch.mm(Q, q_max.transpose(0, 1))
            A = F.softmax(A / scale, dim=0)

            # Bag representation B: (C, D) -> bag logits (C,)
            B = torch.mm(A.transpose(0, 1), V)
            B = B.view(1, B.shape[0], B.shape[1])      # (1, C, D)
            bag_logits_i = self.fcc(B).view(-1)        # (C,)
            patient_bag_logits[p_idx] = bag_logits_i

            # Positive-class attention trace for compatibility
            pos_col = 1 if self.n_classes > 1 else 0
            attention_weights[mask] = A[:, pos_col]
            critical_component_indices[p_idx] = mask.nonzero().flatten()[m_indices[0, pos_col]]

        return patient_bag_logits, patient_max_logits, attention_weights, critical_component_indices

    def forward(self, batch: dict):
        sequences = batch["sequences"]
        masks = batch["masks"]
        component_patient_indices = batch["component_patient_indices"]
        num_patients = int(batch["num_patients"])

        encoded = self.component_encoder(sequences, masks)
        cls_outputs = encoded["cls_output"]   # (N_components, hidden)
        roi_outputs = encoded["roi_outputs"]  # (N_components, MAX_ROIS, hidden)

        component_logits = self.component_classifier(cls_outputs)  # (N_components, C)
        patient_bag_logits, patient_max_logits, attention_weights, critical_component_indices = self._dsmil_pool_per_patient(
            component_embeddings=cls_outputs,
            component_logits=component_logits,
            component_patient_indices=component_patient_indices,
            num_patients=num_patients,
        )

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
            "bag_logits": component_logits,  # component-level logits, used by existing ROI auxiliary branch
            "instance_logits": instance_logits,
            # Keep patient_logits for generic path; DSMIL path should prefer patient_bag_logits
            "patient_logits": patient_bag_logits,
            "patient_bag_logits": patient_bag_logits,
            "patient_max_logits": patient_max_logits,
            "attention_weights": attention_weights,
            "critical_component_indices": critical_component_indices,
        }

