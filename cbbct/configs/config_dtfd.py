"""
DTFD-MIL experiment-only config.

Standalone config to mimic original DTFD-MIL settings while keeping
your default config untouched.
"""

from .config import *  # noqa: F401,F403

# ---------------------------
# Method switch
# ---------------------------
CLS_MODEL_TYPE = "dtfd"

# ---------------------------
# Output isolation
# ---------------------------
OUTPUT_DIR = f"{OUTPUT_DIR}_dtfd"

# ---------------------------
# Original-style DTFD setup
# ---------------------------
# Main_DTFD_MIL.py defaults
DTFD_NUM_GROUP = 4
DTFD_TOTAL_INSTANCE = 4
DTFD_DISTILL_TYPE = "AFS"  # MaxMinS | MaxS | AFS
DTFD_NUM_RES_LAYERS = 0
DTFD_ATTN_DIM = 128
DTFD_DROPRATE_TIER1 = 0.0
DTFD_DROPRATE_TIER2 = 0.0

# Two-tier training loss weights
DTFD_LOSS_TIER1_WEIGHT = 1.0
DTFD_LOSS_TIER2_WEIGHT = 1.0

# Original DTFD commonly uses high hidden dim feature space
# (mDim=512 in repo). Enable this if your memory allows.
# CLS_HIDDEN_DIM = 512

# Optional repo-like optimizer schedule defaults
# CLS_LR = 1e-4
# CLS_EPOCHS = 200
