import scripts._bootstrap  # noqa: F401

import argparse
import os

import config as base_config
import config_camil as camil_config


def _apply_camil_overrides():
    for name in dir(camil_config):
        if name.isupper():
            setattr(base_config, name, getattr(camil_config, name))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run component pipeline with CAMIL."
    )
    parser.add_argument("--roi", type=float, required=True, help="ROI loss weight")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional explicit output dir; default uses config_camil OUTPUT_DIR.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    _apply_camil_overrides()
    base_config.CLS_MODEL_TYPE = "camil"

    if args.output_dir.strip():
        base_config.OUTPUT_DIR = args.output_dir.strip()

    os.makedirs(base_config.OUTPUT_DIR, exist_ok=True)

    import main_component

    run_args = argparse.Namespace(roi=args.roi)
    main_component.main(args=run_args)


if __name__ == "__main__":
    main()
