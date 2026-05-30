"""
MST on ABUS — 官方固定划分 (100/30/70)，depth=32 由 TumorDataset3D 自动 CropOrPad
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 确保能 import mst（无需 pip install，直接 python scripts/main_train_abus.py 即可）
_MST_ROOT = Path(__file__).resolve().parent.parent
if str(_MST_ROOT) not in sys.path:
    sys.path.insert(0, str(_MST_ROOT))

import pandas as pd
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer

from mst.data.datamodules import DataModule
from mst.data.datasets.dataset_3d_abus import DEFAULT_BASE_PATH, get_abus_dataset
from mst.models.dino import DinoV2ClassifierSlice
from mst.models.resnet import ResNet, ResNetSliceTrans

TARGET_SHAPE = (224, 224, 32)


def get_model(name, **kwargs):
    if name == "ResNet":
        return ResNet(in_ch=1, out_ch=2, spatial_dims=3, **kwargs)
    elif name == "ResNetSliceTrans":
        return ResNetSliceTrans(in_ch=1, out_ch=2, spatial_dims=2, **kwargs)
    elif name == "DinoV2ClassifierSlice":
        return DinoV2ClassifierSlice(in_ch=3, out_ch=2, spatial_dims=2, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")


def load_pretrained(model, checkpoint_path: str):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        print(f"No pretrained checkpoint at {checkpoint_path}, training from scratch.")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded pretrained weights from {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MST on ABUS (fixed split)")
    parser.add_argument("--data_root", type=str, default=DEFAULT_BASE_PATH, help="preprocessed_mst root")
    parser.add_argument("--model", type=str, default="DinoV2ClassifierSlice")
    parser.add_argument("--path_root_output", type=str, default="./runs")
    parser.add_argument("--pretrained_ckpt", type=str, default="", help="MST_DUKE.ckpt path (optional)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_time = time.time()
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path(args.path_root_output) / "ABUS" / f"{args.model}_{current_time}"
    path_run_dir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    ds_train = get_abus_dataset("train", base_path=args.data_root, target_shape=TARGET_SHAPE)
    ds_val = get_abus_dataset("val", base_path=args.data_root, target_shape=TARGET_SHAPE)
    ds_test = get_abus_dataset("test", base_path=args.data_root, target_shape=TARGET_SHAPE)

    print(f"ABUS split sizes -> train: {len(ds_train)}, val: {len(ds_val)}, test: {len(ds_test)}")

    train_labels = pd.Series([ds_train.labels_dict[k] for k in ds_train.keys_list])
    class_counts = train_labels.value_counts()
    class_weights = 0.5 / class_counts
    weights = train_labels.map(lambda x: class_weights[x]).values
    weights = torch.tensor(weights, dtype=torch.double)

    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        batch_size=args.batch_size,
        pin_memory=True,
        weights=weights,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = get_model(args.model)
    load_pretrained(model, args.pretrained_ckpt)

    logger = WandbLogger(project="Classifier_ABUS", name=f"ABUS_{args.model}", log_model=False)
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor="val/AUC_ROC",
        save_last=True,
        save_top_k=1,
        mode="max",
        filename="abus-best-{epoch:02d}-{val/AUC_ROC:.4f}",
    )

    trainer = Trainer(
        accelerator=accelerator,
        precision="16-mixed",
        default_root_dir=str(path_run_dir),
        callbacks=[
            checkpointing,
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val/AUC_ROC", patience=30, mode="max"),
        ],
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        max_epochs=args.max_epochs,
        logger=logger,
        num_sanity_val_steps=0,
    )

    print("Starting training...")
    trainer.fit(model, datamodule=dm)

    if checkpointing.best_model_path:
        print(f"Testing with best checkpoint: {checkpointing.best_model_path}")
        trainer.test(model, datamodule=dm, ckpt_path="best")
    else:
        trainer.test(model, datamodule=dm)

    print(f"Total runtime: {time.time() - start_time:.1f}s")
    wandb.finish(quiet=True)
