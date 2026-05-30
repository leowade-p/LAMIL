"""
CLAM experiment-only config.

Standalone config to mimic CLAM default settings without touching
your base config.py.
"""

from .config import *  # noqa: F401,F403

# ---------------------------
# Method switch
# ---------------------------
CLS_MODEL_TYPE = "clam"

# ---------------------------
# Output isolation
# ---------------------------
OUTPUT_DIR = f"{OUTPUT_DIR}_clam"

# ---------------------------
# CLAM-SB style hyper-params
# ---------------------------
CLAM_GATE = True
CLAM_ATTN_HIDDEN_DIM = 256
CLAM_DROPOUT = 0.25
CLAM_K_SAMPLE = 8
CLAM_SUBTYPING = False
CLAM_NO_INST_CLUSTER = False
CLAM_INST_LOSS = "ce"   # ce | svm
CLAM_BAG_WEIGHT = 0.7   # total = bag_weight*bag + (1-bag_weight)*inst
