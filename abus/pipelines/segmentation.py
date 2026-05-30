# segmentation_trainer.py
import logging
import os
from typing import Dict, List

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from dino_finetune import DINOV2EncoderLoRA
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config


class Segmentation2DDataset(Dataset):
    def __init__(self, slice_info_list: List[Dict], dataset_type="train"):
        self.slice_info_list = slice_info_list
        if dataset_type == "train":
            self.transform = A.Compose(
                [
                    A.Rotate(limit=30, p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.RandomScale(scale_limit=0.2, p=0.2),
                    A.LongestMaxSize(max_size=config.SEG_IMG_DIM),
                    A.PadIfNeeded(
                        min_height=config.SEG_IMG_DIM,
                        min_width=config.SEG_IMG_DIM,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                    A.RandomGamma(gamma_limit=(70, 150), p=0.1),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                seed=42,
            )
        else:
            self.transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=config.SEG_IMG_DIM),
                    A.PadIfNeeded(
                        min_height=config.SEG_IMG_DIM,
                        min_width=config.SEG_IMG_DIM,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                seed=42,
            )

    def __len__(self):
        return len(self.slice_info_list)

    def __getitem__(self, idx):
        slice_info = self.slice_info_list[idx]
        img_path = slice_info["image_path"]
        mask_path = slice_info["mask_path"]
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"CV2 failed to read image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            mask = (mask > 127).astype(np.uint8)
            augmented = self.transform(image=img, mask=mask)
            return augmented["image"], augmented["mask"].long()
        except Exception as e:
            logging.error(f"Error processing slice {img_path}: {e}")
            placeholder_img = torch.zeros(3, config.SEG_IMG_DIM, config.SEG_IMG_DIM)
            placeholder_mask = torch.zeros(config.SEG_IMG_DIM, config.SEG_IMG_DIM).long()
            return placeholder_img, placeholder_mask


def dice_loss(pred, target, smooth=1.0):
    pred = torch.softmax(pred, dim=1)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def compute_iou(pred, target, n_classes=config.SEG_N_CLASSES):
    iou_scores = []
    pred = torch.argmax(pred, dim=1)
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).long().sum().item()
        union = (pred_inds | target_inds).long().sum().item()
        iou_scores.append(intersection / union if union > 0 else float("nan"))
    return np.nanmean(iou_scores)


def validate_epoch(model, val_loader, criterion, device, n_classes=config.SEG_N_CLASSES):
    model.eval()
    total_val_loss = 0.0
    total_iou = 0.0
    count = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total_val_loss += criterion(logits, masks).item()
            batch_iou = compute_iou(logits, masks, n_classes)
            if not np.isnan(batch_iou):
                total_iou += batch_iou
            count += 1
    return total_val_loss / count if count > 0 else 0, total_iou / count if count > 0 else 0


def run_segmentation_training(
    fold_identifier: str, train_info: List[Dict], val_info: List[Dict], test_info: List[Dict]
) -> str:
    logging.info(f"========== Starting 2D Segmentation Training ({fold_identifier}) ==========")
    output_dir = os.path.join(config.OUTPUT_DIR, str(fold_identifier))
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_segmentation_model.pt")

    train_loader = DataLoader(
        Segmentation2DDataset(train_info, "train"),
        batch_size=config.SEG_BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        Segmentation2DDataset(val_info, "val"),
        batch_size=config.SEG_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Segmentation2DDataset(test_info, "test"),
        batch_size=config.SEG_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    from dinov2_loader import load_dinov2_base_encoder

    base_encoder = load_dinov2_base_encoder(
        config.DINOV2_REPO_PATH, config.DINOV2_PRETRAINED_WEIGHTS
    )
    model = DINOV2EncoderLoRA(
        encoder=base_encoder,
        r=config.SEG_R_LORA,
        emb_dim=config.SEG_EMB_DIM,
        img_dim=(392, 392),
        n_classes=config.SEG_N_CLASSES,
        use_lora=config.USE_LORA,
        use_fpn=False,
    ).to(config.DEVICE)

    ce_loss = nn.CrossEntropyLoss().to(config.DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.SEG_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.SEG_EPOCHS)

    best_iou = 0.0
    best_epoch = 0
    for epoch in range(config.SEG_EPOCHS):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.SEG_EPOCHS} [Seg Train]")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = 0.6 * ce_loss(logits, masks) + 0.4 * dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss, val_iou = validate_epoch(model, val_loader, ce_loss, config.DEVICE)
        logging.info(
            f"Epoch {epoch + 1}: Train Loss={total_train_loss / len(train_loader):.4f}, "
            f"Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}"
        )
        scheduler.step()
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"  -> New best IoU: {best_iou:.4f}. Saved.")

    model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    _, test_iou = validate_epoch(model, test_loader, ce_loss, config.DEVICE)
    logging.info(f"Segmentation done. Best Val IoU={best_iou:.4f} (epoch {best_epoch}), Test IoU={test_iou:.4f}")
    return best_model_path
