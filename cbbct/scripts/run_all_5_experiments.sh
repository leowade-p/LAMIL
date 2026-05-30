#!/usr/bin/env bash

# 顺序运行 5 个实验脚本：
# 1) ABMIL 2) DSMIL 3) TransMIL 4) DTFD 5) CLAM
#
# 用法示例：
#   chmod +x run_all_5_experiments.sh
#   nohup bash run_all_5_experiments.sh 0 > run_all.log 2>&1 &
#
# 其中第一个参数是 ROI（默认 0）

set -euo pipefail

ROI="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${CBBCT_DIR:-$SCRIPT_DIR}"
DELAY_SECONDS="${DELAY_SECONDS:-0}"

run_one () {
  local name="$1"
  local script="$2"
  local log_file="$3"

  echo "[$(date '+%F %T')] START ${name} (roi=${ROI})"
  (cd "${PROJECT_DIR}" && python "${script}" --roi "${ROI}") > "${PROJECT_DIR}/${log_file}" 2>&1
  echo "[$(date '+%F %T')] DONE  ${name}"
}

echo "[$(date '+%F %T')] DELAY START: waiting 2 hours before running..."
sleep "${DELAY_SECONDS}"
echo "[$(date '+%F %T')] DELAY END: start running experiments."

run_one "ABMIL"   "run_abmil_experiment.py"   "abmil.log"
run_one "DSMIL"   "run_dsmil_experiment.py"   "dsmil.log"
run_one "TransMIL" "run_transmil_experiment.py" "transmil.log"
run_one "DTFD"    "run_dtfd_experiment.py"    "dtfd.log"
run_one "CLAM"    "run_clam_experiment.py"    "clam.log"

echo "[$(date '+%F %T')] ALL DONE"

