"""
TransMIL experiment-only config.

Standalone config for paper arXiv:2106.00908v2 without editing original config.
"""

from .config import *  # noqa: F401,F403

CLS_MODEL_TYPE = "transmil"

# Keep outputs isolated from your default experiments
OUTPUT_DIR = f"{OUTPUT_DIR}_transmil"

