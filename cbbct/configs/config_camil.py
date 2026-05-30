"""
CAMIL experiment-only config.

Standalone config to run CAMIL without editing base config.py.
Ref: CAMIL-main/Camelyon/CAMIL.yaml
"""

from .config import *  # noqa: F401,F403

# ---------------------------
# Method switch
# ---------------------------
CLS_MODEL_TYPE = "camil"

# ---------------------------
# Output isolation
# ---------------------------
OUTPUT_DIR = f"{OUTPUT_DIR}_camil"

# ---------------------------
# CAMIL architecture
# ---------------------------
CAMIL_AGG_DIM = 512
CAMIL_N_LAYERS = 4
CAMIL_TEMPERATURE = 1.2
CAMIL_DROPOUT = 0.15
CAMIL_GATE = True
CAMIL_ATTENTION_DIM = 256

# Optional experiment defaults from official yaml:
# CLS_LR = 2e-4
