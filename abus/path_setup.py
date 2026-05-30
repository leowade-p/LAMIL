"""Ensure abus/ and shared/ are on sys.path before local imports."""
import os
import sys

_ABUS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_ABUS_DIR)
_SHARED_DIR = os.path.join(_REPO_ROOT, "shared")

if _ABUS_DIR in sys.path:
    sys.path.remove(_ABUS_DIR)
sys.path.insert(0, _ABUS_DIR)
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)
