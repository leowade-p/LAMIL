"""Ensure resnet3d baseline, cbbct/, and shared/ are on sys.path."""
import os
import sys

_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_CBBCT_DIR = os.path.dirname(os.path.dirname(_BASELINE_DIR))
_REPO_ROOT = os.path.dirname(_CBBCT_DIR)
_SHARED_DIR = os.path.join(_REPO_ROOT, "shared")

for _path in (_BASELINE_DIR, _CBBCT_DIR, _SHARED_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
