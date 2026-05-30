# Copy to config.py or set environment variables.
import os

DATA_ROOT = os.environ.get("CBBCT_DATA_ROOT", "/path/to/det_dataset")
MASK_DIRS = [
    f"{DATA_ROOT}/panoptic_train2017",
    f"{DATA_ROOT}/panoptic_val2017",
]
WORKING_DIR = os.environ.get("CBBCT_GT_WORKING_DIR", "./outputs")
