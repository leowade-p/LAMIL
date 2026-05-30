"""Ensure cbbct/ and shared/ are on sys.path before local imports."""
import os
import sys

_CBBCT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CBBCT_DIR)
_SHARED_DIR = os.path.join(_REPO_ROOT, "shared")

for _path in (_CBBCT_DIR, _SHARED_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
