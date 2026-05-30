import torch
import torch.nn as nn

from models import ComponentViTEncoder
from rrt_mil_vendor.rrt_model import RRTMIL


def _build_rrtmil_core(
    n_classes: int = 2,
    input_dim: int = 128,
    mlp_dim: int = 512,
    act: str = "relu",
    dropout: float = 0.25,
    region_num: int = 8,
    n_layers: int = 2,
    n_heads: int = 8,
    drop_path: float = 0.0,
    attn: str = "rmsa",
    pool: str = "attn",
    da_act: str = "relu",
    trans_dropout: float = 0.1,
    epeg: bool = True,
    epeg_k: int = 15,
    cr_msa: bool = True,
    crmsa_k: int = 3,
    crmsa_heads: int = 8,
    all_shortcut: bool = False,
    crmsa_mlp: bool = False,
    qkv_bias: bool = True,
    min_region_num: int = 0,
    trans_dim: int = 64,
    ffn: bool = False,
    mlp_ratio: float = 4.0,
) -> RRTMIL:
    return RRTMIL(
        input_dim=input_dim,
        mlp_dim=mlp_dim,
        act=act,
        n_classes=n_classes,
        dropout=dropout,
        pos="none",
        pos_pos=0,
        pool=pool,
        region_num=region_num,
        n_layers=n_layers,
        n_heads=n_heads,
        drop_path=drop_path,
        attn=attn,
        da_act=da_act,
        trans_dropout=trans_dropout,
        ffn=ffn,
        mlp_ratio=mlp_ratio,
        trans_dim=trans_dim,
        epeg=epeg,
        min_region_num=min_region_num,
        qkv_bias=qkv_bias,
        epeg_k=epeg_k,
        cr_msa=cr_msa,
        crmsa_k=crmsa_k,
        crmsa_heads=crmsa_heads,
        all_shortcut=all_shortcut,
        crmsa_mlp=crmsa_mlp,
    )


class RRTMILClassifier(nn.Module):
    """
    RRT-MIL adapted to the current patient-component batch format.

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
        rrt_mlp_dim: int = 512,
        rrt_n_layers: int = 2,
        rrt_n_heads: int = 8,
        rrt_region_num: int = 8,
        rrt_dropout: float = 0.25,
        rrt_act: str = "relu",
        rrt_attn: str = "rmsa",
        rrt_pool: str = "attn",
        rrt_da_act: str = "relu",
        rrt_trans_dropout: float = 0.1,
        rrt_drop_path: float = 0.0,
        rrt_epeg: bool = True,
        rrt_epeg_k: int = 15,
        rrt_cr_msa: bool = True,
        rrt_crmsa_k: int = 3,
        rrt_crmsa_heads: int = 8,
        rrt_all_shortcut: bool = False,
        rrt_crmsa_mlp: bool = False,
        rrt_qkv_bias: bool = True,
        rrt_min_region_num: int = 0,
        rrt_trans_dim: int = 64,
        rrt_ffn: bool = False,
        rrt_mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.component_encoder = ComponentViTEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        self.component_classifier = nn.Linear(hidden_dim, n_classes)
        self.instance_classifier = nn.Linear(hidden_dim, n_classes)
        self.rrt_core = _build_rrtmil_core(
            n_classes=n_classes,
            input_dim=hidden_dim,
            mlp_dim=rrt_mlp_dim,
            act=rrt_act,
            dropout=rrt_dropout,
            region_num=rrt_region_num,
            n_layers=rrt_n_layers,
            n_heads=rrt_n_heads,
            drop_path=rrt_drop_path,
            attn=rrt_attn,
            pool=rrt_pool,
            da_act=rrt_da_act,
            trans_dropout=rrt_trans_dropout,
            epeg=rrt_epeg,
            epeg_k=rrt_epeg_k,
            cr_msa=rrt_cr_msa,
            crmsa_k=rrt_crmsa_k,
            crmsa_heads=rrt_crmsa_heads,
            all_shortcut=rrt_all_shortcut,
            crmsa_mlp=rrt_crmsa_mlp,
            qkv_bias=rrt_qkv_bias,
            min_region_num=rrt_min_region_num,
            trans_dim=rrt_trans_dim,
            ffn=rrt_ffn,
            mlp_ratio=rrt_mlp_ratio,
        )

    def _patient_forward(self, patient_tokens: torch.Tensor) -> torch.Tensor:
        logits = self.rrt_core(patient_tokens)
        if logits.dim() == 2:
            logits = logits.squeeze(0)
        return logits

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
                patient_logits.append(torch.zeros(self.n_classes, device=cls_outputs.device))
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
