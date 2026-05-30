# Copy settings into config.py or set environment variables before running.

import os

#   export ABUS_DATA_ROOT=/path/to/abus
#   export DINOV2_REPO=/path/to/third_party/dinov2
#   export DINOV2_WEIGHTS=/path/to/dinov2_vitl14_reg4_pretrain.pth

DATA_ROOT = os.environ.get("ABUS_DATA_ROOT", "/path/to/abus")
PREPROCESSED_DATA_ROOT = os.path.join(DATA_ROOT, "preprocessed_png_data_1.30")
OUTPUT_DIR = os.path.join(DATA_ROOT, "abus_mil_experiments")
