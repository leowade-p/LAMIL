# mil_model_factory.py - 统一构建各类 MIL 分类器
import logging

import torch.nn as nn

import config
from models_mil import MILClassifier
from models_method_abmil import ABMILClassifier
from models_method_camil import CAMILClassifier
from models_method_clam import CLAMSBClassifier
from models_method_dsmil import DSMILClassifier
from models_method_dtfd import DTFDClassifier
from models_method_rrtmil import RRTMILClassifier
from models_method_transmil import TransMILClassifier


def get_model_type() -> str:
    return getattr(config, "CLS_MODEL_TYPE", "mil_asym").lower()


def build_classifier_model(model_type: str = None) -> nn.Module:
    model_type = (model_type or get_model_type()).lower()

    common = dict(
        input_dim=config.BACKBONE_OUTPUT_DIM,
        hidden_dim=config.CLS_HIDDEN_DIM,
        n_heads=config.CLS_N_HEADS,
        n_layers=config.CLS_N_LAYERS,
    )

    if model_type == "abmil":
        model = ABMILClassifier(
            **common,
            attention_dim=getattr(config, "ABMIL_ATTENTION_DIM", 128),
            gated=bool(getattr(config, "ABMIL_GATED", True)),
        )
        logging.info("Using ABMIL classifier.")
    elif model_type == "dsmil":
        model = DSMILClassifier(
            **common,
            q_dim=int(getattr(config, "DSMIL_Q_DIM", 128)),
            nonlinear=bool(getattr(config, "DSMIL_NONLINEAR", True)),
            passing_v=bool(getattr(config, "DSMIL_PASSING_V", False)),
            dropout_v=float(getattr(config, "DSMIL_DROPOUT_V", 0.0)),
        )
        logging.info("Using DSMIL classifier.")
    elif model_type == "transmil":
        model = TransMILClassifier(**common)
        logging.info("Using TransMIL classifier.")
    elif model_type == "dtfd":
        model = DTFDClassifier(
            **common,
            n_classes=2,
            num_group=int(getattr(config, "DTFD_NUM_GROUP", 4)),
            total_instance=int(getattr(config, "DTFD_TOTAL_INSTANCE", 4)),
            distill_type=str(getattr(config, "DTFD_DISTILL_TYPE", "AFS")),
            num_res_layers=int(getattr(config, "DTFD_NUM_RES_LAYERS", 0)),
            droprate_tier1=float(getattr(config, "DTFD_DROPRATE_TIER1", 0.0)),
            droprate_tier2=float(getattr(config, "DTFD_DROPRATE_TIER2", 0.0)),
            attn_dim=int(getattr(config, "DTFD_ATTN_DIM", 128)),
        )
        logging.info("Using DTFD-MIL classifier.")
    elif model_type == "clam":
        model = CLAMSBClassifier(
            **common,
            n_classes=2,
            gate=bool(getattr(config, "CLAM_GATE", True)),
            attn_hidden_dim=int(getattr(config, "CLAM_ATTN_HIDDEN_DIM", 256)),
            dropout=float(getattr(config, "CLAM_DROPOUT", 0.25)),
            k_sample=int(getattr(config, "CLAM_K_SAMPLE", 8)),
            subtyping=bool(getattr(config, "CLAM_SUBTYPING", False)),
            no_inst_cluster=bool(getattr(config, "CLAM_NO_INST_CLUSTER", False)),
            inst_loss_type=str(getattr(config, "CLAM_INST_LOSS", "ce")),
        )
        logging.info("Using CLAM-SB classifier.")
    elif model_type == "camil":
        model = CAMILClassifier(
            **common,
            n_classes=2,
            agg_dim=int(getattr(config, "CAMIL_AGG_DIM", 512)),
            camil_n_layers=int(getattr(config, "CAMIL_N_LAYERS", 4)),
            temperature=float(getattr(config, "CAMIL_TEMPERATURE", 1.2)),
            dropout=float(getattr(config, "CAMIL_DROPOUT", 0.15)),
            gate=bool(getattr(config, "CAMIL_GATE", True)),
            attention_dim=int(getattr(config, "CAMIL_ATTENTION_DIM", 256)),
        )
        logging.info("Using CAMIL classifier.")
    elif model_type == "rrtmil":
        model = RRTMILClassifier(
            **common,
            n_classes=2,
            rrt_mlp_dim=int(getattr(config, "RRT_MLP_DIM", 512)),
            rrt_n_layers=int(getattr(config, "RRT_N_LAYERS", 2)),
            rrt_n_heads=int(getattr(config, "RRT_N_HEADS", 8)),
            rrt_region_num=int(getattr(config, "RRT_REGION_NUM", 8)),
            rrt_dropout=float(getattr(config, "RRT_DROPOUT", 0.25)),
            rrt_act=str(getattr(config, "RRT_ACT", "relu")),
            rrt_attn=str(getattr(config, "RRT_ATTN", "rmsa")),
            rrt_pool=str(getattr(config, "RRT_POOL", "attn")),
            rrt_da_act=str(getattr(config, "RRT_DA_ACT", "relu")),
            rrt_trans_dropout=float(getattr(config, "RRT_TRANS_DROPOUT", 0.1)),
            rrt_drop_path=float(getattr(config, "RRT_DROP_PATH", 0.0)),
            rrt_epeg=bool(getattr(config, "RRT_EPEG", True)),
            rrt_epeg_k=int(getattr(config, "RRT_EPEG_K", 15)),
            rrt_cr_msa=bool(getattr(config, "RRT_CR_MSA", True)),
            rrt_crmsa_k=int(getattr(config, "RRT_CRMSA_K", 3)),
            rrt_crmsa_heads=int(getattr(config, "RRT_CRMSA_HEADS", 8)),
            rrt_all_shortcut=bool(getattr(config, "RRT_ALL_SHORTCUT", False)),
            rrt_crmsa_mlp=bool(getattr(config, "RRT_CRMSA_MLP", False)),
            rrt_qkv_bias=bool(getattr(config, "RRT_QKV_BIAS", True)),
            rrt_min_region_num=int(getattr(config, "RRT_MIN_REGION_NUM", 0)),
            rrt_trans_dim=int(getattr(config, "RRT_TRANS_DIM", 64)),
            rrt_ffn=bool(getattr(config, "RRT_FFN", False)),
            rrt_mlp_ratio=float(getattr(config, "RRT_MLP_RATIO", 4.0)),
        )
        logging.info("Using RRT-MIL classifier.")
    else:
        model = MILClassifier(**common)
        logging.info("Using asymmetric MIL classifier (mil_asym).")

    return model.to(config.DEVICE)
