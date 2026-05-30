# config.py - ABUS 数据集 MIL 实验基础配置
import os
import sys

_ABUS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.dirname(_ABUS_DIR)
_SHARED_DIR = os.path.join(_REPO_ROOT, "shared")

if _ABUS_DIR in sys.path:
    sys.path.remove(_ABUS_DIR)
sys.path.insert(0, _ABUS_DIR)
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)

import bootstrap  # noqa: F401
from paths import DINOV2_PRETRAINED_WEIGHTS, DINOV2_REPO_PATH, REPO_ROOT, THIRD_PARTY_DIR

import torch

# --- 1. 核心路径配置 ---
# Override via environment: ABUS_DATA_ROOT
DATA_ROOT = os.environ.get("ABUS_DATA_ROOT", "/home/huyiding/pengdie/abus")

TRAIN_DIR_ROOT = os.path.join(DATA_ROOT, "Train")
VAL_DIR_ROOT = os.path.join(DATA_ROOT, "Validation")
TEST_DIR_ROOT = os.path.join(DATA_ROOT, "Test")
PREPROCESSED_DATA_ROOT = os.path.join(DATA_ROOT, "preprocessed_png_data_1.30")

TRAIN_CSV = os.path.join(TRAIN_DIR_ROOT, "labels.csv")
VAL_CSV = os.path.join(VAL_DIR_ROOT, "labels.csv")
TEST_CSV = os.path.join(TEST_DIR_ROOT, "labels.csv")

WORKING_DIR = DATA_ROOT
OUTPUT_DIR = os.path.join(WORKING_DIR, "abus_mil_experiments")
FOLD_IDENTIFIER = "abus_run"

# DINOv2: set DINOV2_REPO / DINOV2_WEIGHTS, or use third_party/dinov2
_default_weights = os.path.join(
    os.environ.get("ABUS_DINOV2_WEIGHTS_DIR", "/home/huyiding/pengdie/dino/dinov2-finetune-main/finalproject"),
    "dinov2_vitl14_reg4_pretrain.pth",
)
if not os.environ.get("DINOV2_WEIGHTS") and os.path.isfile(_default_weights):
    DINOV2_PRETRAINED_WEIGHTS = _default_weights

# 分割模型：若文件已存在则跳过训练
TRAIN_SEGMENTATION = False
SEGMENTATION_MODEL_PATH = ""

# --- 2. 基础配置 ---
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 3. 模型与训练超参数 ---
SEG_IMG_DIM = 392
SEG_EPOCHS = 50
SEG_LR = 3e-4
SEG_BATCH_SIZE = 16
SEG_R_LORA = 3
SEG_EMB_DIM = 1024
SEG_N_CLASSES = 2
USE_LORA = True

MAX_ROIS = 70
ROI_SIZE = 1
DETECTION_THRESHOLD = 0.5
BACKBONE_OUTPUT_DIM = 512

CLS_EPOCHS = 50
CLS_LR = 1e-4
CLS_BATCH_SIZE = 32
CLS_HIDDEN_DIM = 128
CLS_N_HEADS = 4
CLS_N_LAYERS = 2
CLS_POS_EMBED_LEARNABLE = True
CLASS_AUG_RATIO = {0: 5, 1: 5}

CLS_MODEL_TYPE = "mil_asym"

ABMIL_ATTENTION_DIM = 128
ABMIL_GATED = True

DSMIL_NONLINEAR = True
DSMIL_PASSING_V = False
DSMIL_DROPOUT_V = 0.0
DSMIL_Q_DIM = 128
DSMIL_LOSS_BAG_WEIGHT = 0.5
DSMIL_LOSS_MAX_WEIGHT = 0.5
DSMIL_INFER_USE_BAG_LOGITS = True

DTFD_NUM_GROUP = 4
DTFD_TOTAL_INSTANCE = 4
DTFD_DISTILL_TYPE = "AFS"
DTFD_NUM_RES_LAYERS = 0
DTFD_ATTN_DIM = 128
DTFD_DROPRATE_TIER1 = 0.0
DTFD_DROPRATE_TIER2 = 0.0
DTFD_LOSS_TIER1_WEIGHT = 1.0
DTFD_LOSS_TIER2_WEIGHT = 1.0

CLAM_GATE = True
CLAM_ATTN_HIDDEN_DIM = 256
CLAM_DROPOUT = 0.25
CLAM_K_SAMPLE = 8
CLAM_SUBTYPING = False
CLAM_NO_INST_CLUSTER = False
CLAM_INST_LOSS = "ce"
CLAM_BAG_WEIGHT = 0.7

CAMIL_AGG_DIM = 512
CAMIL_N_LAYERS = 4
CAMIL_TEMPERATURE = 1.2
CAMIL_DROPOUT = 0.15
CAMIL_GATE = True
CAMIL_ATTENTION_DIM = 256

RRT_MLP_DIM = 512
RRT_N_LAYERS = 2
RRT_N_HEADS = 8
RRT_REGION_NUM = 8
RRT_DROPOUT = 0.25
RRT_ACT = "relu"
RRT_ATTN = "rmsa"
RRT_POOL = "attn"
RRT_DA_ACT = "relu"
RRT_TRANS_DROPOUT = 0.1
RRT_DROP_PATH = 0.0
RRT_EPEG = True
RRT_EPEG_K = 15
RRT_CR_MSA = True
RRT_CRMSA_K = 3
RRT_CRMSA_HEADS = 8
RRT_ALL_SHORTCUT = False
RRT_CRMSA_MLP = False
RRT_QKV_BIAS = True
RRT_MIN_REGION_NUM = 0
RRT_TRANS_DIM = 64
RRT_FFN = False
RRT_MLP_RATIO = 4.0

MIL_W_PLUS = 5.0
MIL_W_MINUS = 1.0

ALL_PATIENTS_LABELS = {}
