import os
import sys

_GT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _GT_ROOT not in sys.path:
    sys.path.insert(0, _GT_ROOT)

import path_setup  # noqa: F401
