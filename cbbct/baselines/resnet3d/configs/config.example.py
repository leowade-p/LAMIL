import os

DATA_ROOT = os.environ.get(
    "CBBCT_RESNET3D_DATA",
    "/path/to/nnUNet_raw/Dataset510_TumorTotal",
)
OUTPUT_DIR = os.environ.get("CBBCT_RESNET3D_OUTPUT", "./outputs")
