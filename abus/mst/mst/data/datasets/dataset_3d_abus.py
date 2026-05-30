"""
ABUS 3D Dataset for MST — 官方固定划分 Train(100) / Val(30) / Test(70)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from mst.data.datasets.dataset_3d_my import TumorDataset3D

DEFAULT_BASE_PATH = "/home/huyiding/pengdie/abus/preprocessed_mst"


def load_abus_split(base_path: str = DEFAULT_BASE_PATH) -> Tuple[List[str], List[str], List[str], Dict[str, int]]:
    base = Path(base_path)
    split_path = base / "split.json"
    labels_path = base / "classification_labels.json"

    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing {split_path}. Run preprocessing/abus/step1_nrrd_to_nifti_tumor_slices.py first."
        )

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)

    train_keys = split.get("train", [])
    val_keys = split.get("val", [])
    test_keys = split.get("test", [])
    return train_keys, val_keys, test_keys, labels_dict


def get_abus_dataset(
    split: str,
    base_path: str = DEFAULT_BASE_PATH,
    target_shape=(224, 224, 32),
    is_train: bool = None,
):
    train_keys, val_keys, test_keys, labels_dict = load_abus_split(base_path)
    img_dir = os.path.join(base_path, "imagesTr")

    if split == "train":
        keys = train_keys
        is_train = True if is_train is None else is_train
    elif split == "val":
        keys = val_keys
        is_train = False if is_train is None else is_train
    elif split == "test":
        keys = test_keys
        is_train = False if is_train is None else is_train
    else:
        raise ValueError(f"Unknown split: {split}")

    return TumorDataset3D(
        img_dir=img_dir,
        keys_list=keys,
        labels_dict=labels_dict,
        target_shape=target_shape,
        is_train=is_train,
    )
