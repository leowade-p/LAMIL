import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ComponentViTEncoder


class AttnNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x), x


class AttnNetGated(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        a_layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        b_layers = [nn.Linear(in_dim, hidden_dim), nn.Sigmoid()]
        if dropout > 0:
            a_layers.append(nn.Dropout(dropout))
            b_layers.append(nn.Dropout(dropout))
        self.attention_a = nn.Sequential(*a_layers)
        self.attention_b = nn.Sequential(*b_layers)
        self.attention_c = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)
        return A, x


class CLAMSBClassifier(nn.Module):
    """
    CLAM-SB adapted to the current patient-component batch format.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_classes: int = 2,
        gate: bool = True,
        attn_hidden_dim: int = 256,
        dropout: float = 0.25,
        k_sample: int = 8,
        subtyping: bool = False,
        no_inst_cluster: bool = False,
        inst_loss_type: str = "ce",
    ):
        super().__init__()
        self.component_encoder = ComponentViTEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )

        self.n_classes = n_classes
        self.k_sample = k_sample
        self.subtyping = subtyping
        self.no_inst_cluster = no_inst_cluster

        self.pre_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_net = AttnNetGated(hidden_dim, attn_hidden_dim, dropout=dropout) if gate else AttnNet(hidden_dim, attn_hidden_dim, dropout=dropout)
        self.bag_classifier = nn.Linear(hidden_dim, n_classes)
        self.instance_classifiers = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(n_classes)])

        # Compatibility outputs for existing auxiliary branch
        self.component_classifier = nn.Linear(hidden_dim, n_classes)
        self.instance_classifier = nn.Linear(hidden_dim, n_classes)

        self.instance_loss_fn = nn.CrossEntropyLoss() if inst_loss_type != "svm" else nn.CrossEntropyLoss()

    @staticmethod
    def _create_pos_targets(length: int, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def _create_neg_targets(length: int, device):
        return torch.full((length,), 0, device=device).long()

    def _inst_eval(self, A: torch.Tensor, h: torch.Tensor, classifier: nn.Module):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k = min(self.k_sample, A.shape[-1])
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)

        p_targets = self._create_pos_targets(k, device)
        n_targets = self._create_neg_targets(k, device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        inst_loss = self.instance_loss_fn(logits, all_targets)
        return inst_loss, all_preds, all_targets

    def _inst_eval_out(self, A: torch.Tensor, h: torch.Tensor, classifier: nn.Module):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k = min(self.k_sample, A.shape[-1])
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self._create_neg_targets(k, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        inst_loss = self.instance_loss_fn(logits, p_targets)
        return inst_loss, p_preds, p_targets

    def _forward_one_patient(self, h: torch.Tensor, label: torch.Tensor):
        h = self.pre_fc(h)
        A, h = self.attention_net(h)   # (N, 1), (N, D)
        A = torch.transpose(A, 1, 0)   # (1, N)
        A_raw = A
        A = F.softmax(A, dim=1)

        M = torch.mm(A, h)             # (1, D)
        bag_logit = self.bag_classifier(M).squeeze(0)  # (C,)

        total_inst_loss = torch.tensor(0.0, device=h.device)
        inst_preds = []
        inst_targets = []
        inst_count = 0

        if (not self.no_inst_cluster) and label is not None:
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze(0)
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    inst_loss, preds, targets = self._inst_eval(A, h, classifier)
                    inst_preds.extend(preds.detach().cpu().numpy())
                    inst_targets.extend(targets.detach().cpu().numpy())
                    total_inst_loss = total_inst_loss + inst_loss
                    inst_count += 1
                elif self.subtyping:
                    inst_loss, preds, targets = self._inst_eval_out(A, h, classifier)
                    inst_preds.extend(preds.detach().cpu().numpy())
                    inst_targets.extend(targets.detach().cpu().numpy())
                    total_inst_loss = total_inst_loss + inst_loss
                    inst_count += 1

            if self.subtyping and inst_count > 0:
                total_inst_loss = total_inst_loss / inst_count

        return bag_logit, A_raw, total_inst_loss, inst_preds, inst_targets

    def forward(self, batch: dict):
        sequences = batch["sequences"]
        masks = batch["masks"]
        component_patient_indices = batch["component_patient_indices"]
        num_patients = int(batch["num_patients"])
        labels = batch.get("labels")

        encoded = self.component_encoder(sequences, masks)
        cls_outputs = encoded["cls_output"]   # (N_components, D)
        roi_outputs = encoded["roi_outputs"]  # (N_components, MAX_ROIS, D)

        # Compatibility output (component-level)
        bag_logits_components = self.component_classifier(cls_outputs)

        patient_logits = []
        all_inst_loss = torch.tensor(0.0, device=cls_outputs.device)
        inst_eval_count = 0
        all_inst_preds = []
        all_inst_targets = []

        for p_idx in range(num_patients):
            p_mask = component_patient_indices == p_idx
            if not torch.any(p_mask):
                patient_logits.append(torch.zeros(self.n_classes, device=cls_outputs.device))
                continue

            h = cls_outputs[p_mask]
            label = labels[p_idx].view(1) if labels is not None else None
            logit, _Araw, inst_loss, inst_preds, inst_targets = self._forward_one_patient(h, label)
            patient_logits.append(logit)

            if (not self.no_inst_cluster) and label is not None:
                all_inst_loss = all_inst_loss + inst_loss
                inst_eval_count += 1
                all_inst_preds.extend(inst_preds)
                all_inst_targets.extend(inst_targets)

        patient_logits = torch.stack(patient_logits, dim=0) if patient_logits else torch.empty(0, self.n_classes, device=cls_outputs.device)
        clam_instance_loss = all_inst_loss / inst_eval_count if inst_eval_count > 0 else torch.tensor(0.0, device=cls_outputs.device)

        # ROI-level compatibility output
        bsz, seq_len, _ = roi_outputs.shape
        if masks.shape[1] != seq_len:
            if masks.shape[1] > seq_len:
                masks = masks[:, :seq_len]
            else:
                pad = torch.zeros(bsz, seq_len - masks.shape[1], dtype=torch.bool, device=masks.device)
                masks = torch.cat([masks, pad], dim=1)
        valid_roi_outputs = roi_outputs[masks]
        instance_logits = self.instance_classifier(valid_roi_outputs)

        return {
            "bag_logits": bag_logits_components,
            "instance_logits": instance_logits,
            "patient_logits": patient_logits,
            "clam_instance_loss": clam_instance_loss,
            "clam_inst_preds": np.array(all_inst_preds),
            "clam_inst_targets": np.array(all_inst_targets),
        }

