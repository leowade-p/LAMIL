import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ComponentViTEncoder


class Classifier1FC(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, droprate: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
        self.dropout = nn.Dropout(droprate) if droprate > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DimReduction(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_res_layers: int = 0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.num_res_layers = num_res_layers
        if num_res_layers > 0:
            self.res_blocks = nn.Sequential(*[ResidualBlock(out_dim) for _ in range(num_res_layers)])
        else:
            self.res_blocks = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc(x))
        if self.res_blocks is not None:
            x = self.res_blocks(x)
        return x


class AttentionGated(nn.Module):
    def __init__(self, feat_dim: int, attn_dim: int = 128, k: int = 1):
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attn_dim, k)

    def forward(self, x: torch.Tensor, is_norm: bool = True) -> torch.Tensor:
        # x: (N, D)
        a_v = self.attention_v(x)
        a_u = self.attention_u(x)
        a = self.attention_w(a_v * a_u)  # (N, K)
        a = torch.transpose(a, 1, 0)  # (K, N)
        if is_norm:
            a = F.softmax(a, dim=1)
        return a


class AttentionWithClassifier(nn.Module):
    def __init__(self, feat_dim: int, n_classes: int, attn_dim: int = 128, droprate: float = 0.0):
        super().__init__()
        self.attention = AttentionGated(feat_dim, attn_dim=attn_dim, k=1)
        self.classifier = Classifier1FC(feat_dim, n_classes, droprate=droprate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D)
        a = self.attention(x)       # (1, N)
        feat = torch.mm(a, x)       # (1, D)
        pred = self.classifier(feat)  # (1, C)
        return pred


class DTFDClassifier(nn.Module):
    """
    DTFD-MIL style two-tier classifier adapted to current project batch format.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_classes: int = 2,
        num_group: int = 4,
        total_instance: int = 4,
        distill_type: str = "AFS",
        num_res_layers: int = 0,
        droprate_tier1: float = 0.0,
        droprate_tier2: float = 0.0,
        attn_dim: int = 128,
    ):
        super().__init__()
        self.component_encoder = ComponentViTEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        self.num_group = max(1, int(num_group))
        self.total_instance = max(1, int(total_instance))
        self.distill_type = distill_type
        self.n_classes = n_classes

        self.dim_reduction = DimReduction(hidden_dim, hidden_dim, num_res_layers=num_res_layers)
        self.attention = AttentionGated(hidden_dim, attn_dim=attn_dim, k=1)
        self.classifier_tier1 = Classifier1FC(hidden_dim, n_classes, droprate=droprate_tier1)
        self.classifier_tier2 = AttentionWithClassifier(hidden_dim, n_classes, attn_dim=attn_dim, droprate=droprate_tier2)

        # For compatibility with existing auxiliary branch
        self.component_classifier = nn.Linear(hidden_dim, n_classes)
        self.instance_classifier = nn.Linear(hidden_dim, n_classes)

    def _cam_1d(self, features: torch.Tensor) -> torch.Tensor:
        # features: (N, D) -> return (N, C) logits
        weight = self.classifier_tier1.fc.weight  # (C, D)
        cam = torch.einsum("nd,cd->nc", features, weight)
        return cam

    def _split_indices(self, n: int) -> List[List[int]]:
        idx = list(range(n))
        random.shuffle(idx)
        chunks = np.array_split(np.array(idx), self.num_group)
        out = [c.tolist() for c in chunks if len(c) > 0]
        return out if out else [idx]

    def _distill_group_features(self, mid_feat: torch.Tensor, att_feat: torch.Tensor) -> torch.Tensor:
        # mid_feat: (Ng, D), att_feat: (1, D)
        inst_per_group = max(1, self.total_instance // self.num_group)
        if self.distill_type == "AFS":
            return att_feat

        patch_logits = self._cam_1d(mid_feat)  # (Ng, C)
        patch_prob = torch.softmax(patch_logits, dim=1)
        _, sort_idx = torch.sort(patch_prob[:, -1], descending=True)

        if self.distill_type == "MaxS":
            topk_idx = sort_idx[: min(inst_per_group, len(sort_idx))].long()
            return mid_feat.index_select(dim=0, index=topk_idx)

        # Default/MaxMinS
        topk_max = sort_idx[: min(inst_per_group, len(sort_idx))].long()
        topk_min = sort_idx[-min(inst_per_group, len(sort_idx)) :].long()
        topk = torch.cat([topk_max, topk_min], dim=0)
        return mid_feat.index_select(dim=0, index=topk)

    def forward(self, batch: dict):
        sequences = batch["sequences"]
        masks = batch["masks"]
        component_patient_indices = batch["component_patient_indices"]
        num_patients = int(batch["num_patients"])

        encoded = self.component_encoder(sequences, masks)
        cls_outputs = encoded["cls_output"]   # (N_components, hidden)
        roi_outputs = encoded["roi_outputs"]  # (N_components, MAX_ROIS, hidden)

        # Compatibility outputs
        bag_logits = self.component_classifier(cls_outputs)

        patient_logits = []
        tier1_logits_all = []
        tier1_patient_indices = []

        for p_idx in range(num_patients):
            p_mask = component_patient_indices == p_idx
            if not torch.any(p_mask):
                patient_logits.append(torch.zeros(self.n_classes, device=cls_outputs.device))
                continue

            feat = cls_outputs[p_mask]  # (K, D)
            group_indices = self._split_indices(feat.shape[0])
            pseudo_feat = []

            for g in group_indices:
                g_idx = torch.tensor(g, dtype=torch.long, device=feat.device)
                g_feat = feat.index_select(dim=0, index=g_idx)  # (Ng, D)
                mid_feat = self.dim_reduction(g_feat)           # (Ng, D)

                a = self.attention(mid_feat, is_norm=False).squeeze(0)  # (Ng,)
                a = torch.softmax(a, dim=0)
                att_feats = torch.einsum("nd,n->nd", mid_feat, a)
                att_feat = torch.sum(att_feats, dim=0, keepdim=True)  # (1, D)

                pred_tier1 = self.classifier_tier1(att_feat)  # (1, C)
                tier1_logits_all.append(pred_tier1)
                tier1_patient_indices.append(p_idx)

                d_feat = self._distill_group_features(mid_feat, att_feat)
                pseudo_feat.append(d_feat)

            slide_pseudo_feat = torch.cat(pseudo_feat, dim=0)   # (Ndistill, D)
            pred_tier2 = self.classifier_tier2(slide_pseudo_feat).squeeze(0)  # (C,)
            patient_logits.append(pred_tier2)

        patient_logits = torch.stack(patient_logits, dim=0) if patient_logits else torch.empty(0, self.n_classes, device=cls_outputs.device)
        if tier1_logits_all:
            tier1_logits = torch.cat(tier1_logits_all, dim=0)  # (sum_groups, C)
            tier1_patient_indices = torch.tensor(tier1_patient_indices, dtype=torch.long, device=cls_outputs.device)
        else:
            tier1_logits = torch.empty(0, self.n_classes, device=cls_outputs.device)
            tier1_patient_indices = torch.empty(0, dtype=torch.long, device=cls_outputs.device)

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
            "bag_logits": bag_logits,
            "instance_logits": instance_logits,
            "patient_logits": patient_logits,  # second tier
            "tier1_logits": tier1_logits,      # first tier group-level
            "tier1_patient_indices": tier1_patient_indices,
        }

