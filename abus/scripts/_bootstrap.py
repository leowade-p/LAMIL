"""Ensure modality root and shared/ are on sys.path when running from scripts/."""
import os
import sys

_MODALITY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MODALITY_ROOT not in sys.path:
    sys.path.insert(0, _MODALITY_ROOT)

import path_setup  # noqa: F401
