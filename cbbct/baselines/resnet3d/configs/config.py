"""3D ResNet baseline configuration."""
import os
import sys

_BASELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CBBCT_DIR = os.path.dirname(os.path.dirname(_BASELINE_DIR))
_REPO_ROOT = os.path.dirname(_CBBCT_DIR)
if _BASELINE_DIR not in sys.path:
    sys.path.insert(0, _BASELINE_DIR)

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_ROOT = os.environ.get(
    "CBBCT_RESNET3D_DATA",
    "/home/huyiding/pengdie/tumor_classification/nnUNet_raw/Dataset510_TumorTotal",
)
OUTPUT_DIR = os.environ.get(
    "CBBCT_RESNET3D_OUTPUT",
    os.path.join(_BASELINE_DIR, "outputs"),
)

TARGET_SIZE = 392
TEMPORAL_SIZE = 160
RANDOM_STATE = 42
K_FOLDS = 5
NUM_EPOCHS = 300
NUM_CLASSES = 2
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

KINETICS_MEAN = [0.43216, 0.39466, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

# Patient labels are defined in dataloader/dataset.py (ALL_PATIENTS_LABELS)
