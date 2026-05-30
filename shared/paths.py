"""Repository-wide path helpers for DINOv2 and shared assets."""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARED_DIR = os.path.join(REPO_ROOT, "shared")
THIRD_PARTY_DIR = os.path.join(REPO_ROOT, "third_party")

DINOV2_REPO_PATH = os.environ.get(
    "DINOV2_REPO",
    os.path.join(THIRD_PARTY_DIR, "dinov2"),
)
DINOV2_PRETRAINED_WEIGHTS = os.environ.get(
    "DINOV2_WEIGHTS",
    os.path.join(THIRD_PARTY_DIR, "dinov2_vitl14_reg4_pretrain.pth"),
)
