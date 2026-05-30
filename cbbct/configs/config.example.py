# Copy to config.py and edit paths for your machine.
# Patient fold labels are already in config.py (ALL_PATIENTS_LABELS).

import os

# Environment overrides (recommended):
#   export CBBCT_DATA_ROOT=/path/to/det_datasets/4v1withouts1s2
#   export CBBCT_WORKING_DIR=/path/to/experiment_outputs
#   export DINOV2_REPO=/path/to/third_party/dinov2
#   export DINOV2_WEIGHTS=/path/to/dinov2_vitl14_reg4_pretrain.pth

DATA_ROOT = os.environ.get("CBBCT_DATA_ROOT", "/path/to/cbbct/det_dataset")
WORKING_DIR = os.environ.get("CBBCT_WORKING_DIR", "/path/to/cbbct/workdir")
OUTPUT_DIR = os.path.join(WORKING_DIR, "kfold_results")
SEG_MODEL_PATH_TEMPLATE = os.path.join(
    WORKING_DIR, "kfold_results", "fold_{fold_num}", "best_segmentation_model.pt"
)
