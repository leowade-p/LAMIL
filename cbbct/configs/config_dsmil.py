"""
DSMIL experiment-only config.

Standalone config for paper arXiv:2011.08939v3 without editing original config.
"""

from .config import *  # noqa: F401,F403

# ---------------------------
# Method switch
# ---------------------------
CLS_MODEL_TYPE = "dsmil"

# ---------------------------
# Output isolation
# ---------------------------
OUTPUT_DIR = f"{OUTPUT_DIR}_dsmil"

# ---------------------------
# DSMIL architecture (official-style)
# Ref: dsmil.py -> BClassifier(...)
# ---------------------------
# Whether to use nonlinear q(.) branch:
# True: Linear -> ReLU -> Linear -> Tanh (official default)
# False: single Linear
DSMIL_NONLINEAR = True

# Whether to add v(.) projection with Dropout + Linear + ReLU.
# Official default is False (Identity).
DSMIL_PASSING_V = False
DSMIL_DROPOUT_V = 0.0

# q feature dimension in official implementation is fixed to 128.
DSMIL_Q_DIM = 128

# ---------------------------
# DSMIL training objective (official-style)
# Ref: train_mil.py:
#   loss_total = 0.5 * loss_bag + 0.5 * loss_max
# ---------------------------
DSMIL_LOSS_BAG_WEIGHT = 0.5
DSMIL_LOSS_MAX_WEIGHT = 0.5

# Use bag branch as final probability at inference time
# (consistent with official testing scripts).
DSMIL_INFER_USE_BAG_LOGITS = True

# ---------------------------
# Optional DSMIL experiment defaults
# ---------------------------
# Keep project defaults unless explicitly changed.
# Uncomment if you want to match train_mil.py closer:
# CLS_LR = 2e-4
# CLS_EPOCHS = 40
# WEIGHT_DECAY = 5e-3

