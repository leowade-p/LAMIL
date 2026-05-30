"""
ABMIL experiment-only config.

This file does NOT replace your original `config.py`.
Use it with `run_abmil_experiment.py` to run ABMIL without
editing your existing training pipeline files.
"""

from .config import *  # noqa: F401,F403

# ---------------------------
# Method switch
# ---------------------------
CLS_MODEL_TYPE = "abmil"   # your pipeline already supports this switch
ABMIL_ATTENTION_DIM = 128
ABMIL_GATED = True

# ---------------------------
# Experiment output isolation
# ---------------------------
# Keep original outputs untouched by writing to a sibling directory.
OUTPUT_DIR = f"{OUTPUT_DIR}_abmil"

