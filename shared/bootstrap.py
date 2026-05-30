"""Add shared/ to sys.path so top-level imports resolve consistently."""
import os
import sys

_SHARED_DIR = os.path.dirname(os.path.abspath(__file__))
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)
