# run_all_mil_experiments.py - 批量运行 ABUS 全部 MIL 对比实验
import scripts._bootstrap  # noqa: F401
"""
用法:
    cd abus
    python run_all_mil_experiments.py --roi 0.0
    python run_all_mil_experiments.py --roi 1.0 --methods abmil dsmil transmil
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime

ABUS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ABUS_DIR, "scripts")

EXPERIMENTS = [
    ("run_mil_asym_experiment.py", "mil_asym"),
    ("run_abmil_experiment.py", "abmil"),
    ("run_dsmil_experiment.py", "dsmil"),
    ("run_transmil_experiment.py", "transmil"),
    ("run_dtfd_experiment.py", "dtfd"),
    ("run_clam_experiment.py", "clam"),
    ("run_camil_experiment.py", "camil"),
    ("run_rrtmil_experiment.py", "rrtmil"),
]


def main():
    parser = argparse.ArgumentParser(description="Run all ABUS MIL experiments sequentially")
    parser.add_argument("--roi", type=float, default=0.0, help="ROI loss weight for all runs")
    parser.add_argument("--w-plus", type=float, default=None, help="mil_asym positive weight")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[name for _, name in EXPERIMENTS],
        help="Subset of methods to run",
    )
    parser.add_argument("--train-seg", action="store_true", help="Retrain segmentation for each run")
    args = parser.parse_args()

    selected = {m.lower() for m in args.methods}
    log_dir = os.path.join(ABUS_DIR, "batch_logs")
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    results = []
    for script, method in EXPERIMENTS:
        if method not in selected:
            continue
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, script), "--roi", str(args.roi)]
        if args.w_plus is not None:
            cmd.extend(["--w-plus", str(args.w_plus)])
        if args.train_seg:
            cmd.append("--train-seg")

        print(f"\n{'=' * 60}\nRunning {method}: {' '.join(cmd)}\n{'=' * 60}")
        log_file = os.path.join(log_dir, f"{method}.log")
        with open(log_file, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, cwd=ABUS_DIR, stdout=f, stderr=subprocess.STDOUT)
        status = "OK" if proc.returncode == 0 else f"FAILED({proc.returncode})"
        results.append((method, status, log_file))
        print(f"{method}: {status} -> log: {log_file}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"ROI={args.roi}\n")
        for method, status, log_file in results:
            f.write(f"{method}\t{status}\t{log_file}\n")
    print(f"\nBatch summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
