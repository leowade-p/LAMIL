"""
RRT-MIL experiment-only config.

Standalone config to run RRT-MIL without editing base config.py.
Ref: RRT-MIL-master/main.py (Camelyon16-R50 defaults)
"""

from .config import *  # noqa: F401,F403

# ---------------------------
# Method switch
# ---------------------------
CLS_MODEL_TYPE = "rrtmil"

# ---------------------------
# Output isolation
# ---------------------------
OUTPUT_DIR = f"{OUTPUT_DIR}_rrtmil"

# ---------------------------
# RRT-MIL architecture
# ---------------------------
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

# Optional experiment defaults from official script:
# CLS_LR = 2e-4
