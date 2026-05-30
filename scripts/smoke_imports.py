#!/usr/bin/env python3
"""Syntax and optional import smoke tests for the reorganized repo."""
import compileall
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def compile_packages() -> bool:
    ok = True
    for target in (
        os.path.join(REPO_ROOT, "shared"),
        os.path.join(REPO_ROOT, "cbbct"),
        os.path.join(REPO_ROOT, "cbbct", "gt"),
        os.path.join(REPO_ROOT, "cbbct", "baselines", "resnet3d"),
    ):
        print("compileall: {}".format(target))
        if not compileall.compile_dir(target, quiet=1):
            ok = False

    abus_dir = os.path.join(REPO_ROOT, "abus")
    print("compileall: {} (top-level .py only)".format(abus_dir))
    for name in os.listdir(abus_dir):
        if name.endswith(".py"):
            path = os.path.join(abus_dir, name)
            if not compileall.compile_file(path, quiet=1):
                ok = False
    return ok


def try_imports() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP import tests: torch not installed")
        return True

    sys.path.insert(0, os.path.join(REPO_ROOT, "cbbct"))
    sys.path.insert(0, os.path.join(REPO_ROOT, "abus"))
    sys.path.insert(0, os.path.join(REPO_ROOT, "shared"))

    sys.path.insert(0, os.path.join(REPO_ROOT, "cbbct"))
    import path_setup  # noqa: F401

    os.chdir(os.path.join(REPO_ROOT, "cbbct"))
    import config as cbbct_config  # noqa: F401

    print("OK: cbbct.config", cbbct_config.DINOV2_REPO_PATH)

    os.chdir(os.path.join(REPO_ROOT, "abus"))
    import importlib

    importlib.invalidate_caches()
    if "config" in sys.modules:
        del sys.modules["config"]
    sys.path.insert(0, os.path.join(REPO_ROOT, "abus"))
    import path_setup  # noqa: F401
    import config as abus_config  # noqa: F401

    print("OK: abus.config", abus_config.DATA_ROOT)
    return True


def main() -> int:
    os.chdir(REPO_ROOT)
    ok = compile_packages()
    ok = try_imports() and ok
    if not ok:
        print("SMOKE TEST FAILED")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
