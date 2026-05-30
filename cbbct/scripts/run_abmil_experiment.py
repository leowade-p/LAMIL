import scripts._bootstrap  # noqa: F401

import argparse
import os

import config as base_config
import config_abmil as abmil_config


def _apply_abmil_overrides():
    """
    Runtime override: copy UPPER_CASE attrs from config_abmil -> config.
    This avoids editing your original config.py and keeps old scripts intact.
    """
    for name in dir(abmil_config):
        if name.isupper():
            setattr(base_config, name, getattr(abmil_config, name))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run component pipeline with ABMIL without touching original code."
    )
    parser.add_argument("--roi", type=float, required=True, help="ROI loss weight")
    parser.add_argument(
        "--gated",
        type=int,
        default=1,
        choices=[0, 1],
        help="1=gated attention, 0=plain attention",
    )
    parser.add_argument(
        "--attention-dim",
        type=int,
        default=128,
        help="ABMIL attention hidden dimension",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional explicit output dir. If empty, uses config_abmil OUTPUT_DIR.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # 1) apply ABMIL defaults
    _apply_abmil_overrides()

    # 2) apply cli overrides
    base_config.ABMIL_GATED = bool(args.gated)
    base_config.ABMIL_ATTENTION_DIM = int(args.attention_dim)
    base_config.CLS_MODEL_TYPE = "abmil"

    if args.output_dir.strip():
        base_config.OUTPUT_DIR = args.output_dir.strip()

    os.makedirs(base_config.OUTPUT_DIR, exist_ok=True)

    # 3) reuse your existing pipeline entry; no source file edits needed
    import main_component

    run_args = argparse.Namespace(roi=args.roi)
    main_component.main(args=run_args)


if __name__ == "__main__":
    main()

