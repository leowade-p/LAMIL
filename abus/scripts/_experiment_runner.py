"""ABUS MIL 实验通用启动器。"""
import argparse
import importlib
import os
import sys

# 本文件所在目录 = abus/（MIL 实验代码目录）
_ABUS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.dirname(_ABUS_DIR)
_SHARED_DIR = os.path.join(_REPO_ROOT, "shared")

# 本地实验模块名：必须与 abus/abus/ 下的 .py 文件对应
_LOCAL_MODULES = (
    "config",
    "main_component",
    "utils",
    "data_loader",
    "classification_pipeline",
    "segmentation_trainer",
    "mil_model_factory",
    "mil_eval",
)


def _setup_import_paths():
    """确保 abus/ 与 shared/ 在 sys.path 中且 abus 模块优先。"""
    if _ABUS_DIR in sys.path:
        sys.path.remove(_ABUS_DIR)
    sys.path.insert(0, _ABUS_DIR)

    if _SHARED_DIR not in sys.path:
        sys.path.insert(0, _SHARED_DIR)

    # 清除已从父目录 abus/ 误加载的同名模块缓存
    for name in _LOCAL_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", None)
        if mod_file is None:
            del sys.modules[name]
            continue
        mod_dir = os.path.dirname(os.path.abspath(mod_file))
        if mod_dir != _ABUS_DIR:
            del sys.modules[name]


def _import_local(module_name: str):
    _setup_import_paths()
    return importlib.import_module(module_name)


def run_experiment(method_config_module: str, model_type: str, extra_cli=None):
    parser = argparse.ArgumentParser(description=f"ABUS MIL experiment: {model_type}")
    parser.add_argument("--roi", type=float, default=0.0, help="ROI loss weight")
    parser.add_argument("--w-plus", type=float, default=None, help="mil_asym positive weight")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output directory override")
    parser.add_argument("--train-seg", action="store_true", help="Force retrain segmentation model")
    if extra_cli:
        extra_cli(parser)
    args = parser.parse_args()

    _setup_import_paths()

    base_config = _import_local("config")
    method_cfg = importlib.import_module(method_config_module)
    for name in dir(method_cfg):
        if name.isupper():
            setattr(base_config, name, getattr(method_cfg, name))
    base_config.CLS_MODEL_TYPE = model_type
    if args.output_dir.strip():
        base_config.OUTPUT_DIR = args.output_dir.strip()
    os.makedirs(base_config.OUTPUT_DIR, exist_ok=True)

    main_component = _import_local("main_component")
    main_component.main(args)


if __name__ == "__main__":
    raise SystemExit("Use run_*_experiment.py instead.")
