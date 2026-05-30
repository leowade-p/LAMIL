#!/bin/bash
# ABUS MST 完整流程：预处理 + 训练

set -e
export WANDB_MODE=offline

# 修改为你的 ABUS 根目录
DATA_ROOT="${DATA_ROOT:-/home/huyiding/pengdie/abus}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/home/huyiding/pengdie/mst/MST_DUKE.ckpt}"

MST_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${MST_ROOT}:${PYTHONPATH}"
cd "$MST_ROOT"

echo "MST_ROOT=$MST_ROOT"
echo "PYTHONPATH=$PYTHONPATH"

echo "========== Step 1: NRRD -> NIfTI (tumor slices only) =========="
python scripts/preprocessing/abus/step1_nrrd_to_nifti_tumor_slices.py \
    --data_root "$DATA_ROOT" \
    --skip_existing

echo "========== Step 2: Train MST on ABUS =========="
python scripts/main_train_abus.py \
    --data_root "$DATA_ROOT/preprocessed_mst" \
    --pretrained_ckpt "$PRETRAINED_CKPT" \
    --path_root_output ./runs

echo "Done."