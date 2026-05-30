import os
import cv2
import torch
import numpy as np
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torchvision.ops import roi_align
from typing import List, Dict, Any
import tqdm
import config
import models
import utils
from collections import defaultdict
class ThreeDeeOnlineGTDataset(Dataset):
    def __init__(self, image_dir: List[str], vit_pids: List[str],mask_dir: List[str], is_train: bool):
        """
        适用于 2D PNG 图像和掩码的数据加载器。

        Args:
            image_dir (str): 图像文件所在目录。
            mask_dir (str): 掩码文件所在目录。
            is_train (bool): 是否为训练模式，决定是否应用数据增强。
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.pids=vit_pids
        
        # --- Backbone 初始化 ---
        # 加载预训练的 ResNet50，并移除最后的 average pooling 和全连接层
        # 我们需要 C4 或 C5 层的输出，这里以 C4 (layer3) 为例，输出通道数为 1024
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.feature_backbone = nn.Sequential(*list(resnet.children())[:-4]) # 输出 C4 特征图
        self.feature_backbone = self.feature_backbone.to(config.DEVICE)
        self.feature_backbone.eval()
        
        # 冻结所有参数，因为我们只用它来提取特征
        for param in self.feature_backbone.parameters():
            param.requires_grad = False
            
        # --- 数据增强和预处理流水线 ---
        if self.is_train:
            self.dataset_type='train'
            # 训练时应用数据增强
            self.transform = A.Compose([
                A.Resize(392, 392, interpolation=cv2.INTER_LANCZOS4),
                A.Rotate(limit=15, p=0.1, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.RandomResizedCrop(height=392, width=392, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.15),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'})
        else:
            self.dataset_type='valortest'
            # 验证/测试时不应用数据增强，只有缩放和归一化
            self.transform = A.Compose([
                A.Resize(392, 392, interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'})
        

    def __len__(self):
        return len(self.image_files)
    def __getitem__(self,idx):
        pid = self.pids[idx]
        self.imagefiles=utils.get_image_files_for_pids(self.image_dir, pid)
        self.maskfiles=utils.get_image_files_for_pids(self.mask_dir, pid)
        num_rois, C, H, W = self.feature_backbones.shape
        progress_bar = tqdm(self.imagefiles, desc=f"Extracting Features ({self.dataset_type})")
        self.patient_bank = defaultdict(list)
        for img_path in progress_bar:
            pid_prefix = os.path.basename(img_path).split('_')[0]
            
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
            mask = Image.open(img_path).convert("L")
            mask_np = np.array(mask)

            # Pass the NumPy array to the transform. Albumentations returns a dictionary.
            transformed = self.transform(image=image_np,mask=mask_np)
            # Get the transformed image tensor from the dictionary
            tensor = transformed['image'].unsqueeze(0).to(config.DEVICE)
            mask=transformed['mask']

            with torch.no_grad():
                feat_map = self.feature_backbone(tensor)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            spatial_scale_x = feat_map.shape[3] / tensor.shape[3]
            spatial_scale_y = feat_map.shape[2] / tensor.shape[2]
            
            boxes_scaled = [
                [
                    c[0] * spatial_scale_x, 
                    c[1] * spatial_scale_y, 
                    (c[0] + c[2]) * spatial_scale_x, 
                    (c[1] + c[3]) * spatial_scale_y
                ]
                for cnt in contours for c in [cv2.boundingRect(cnt)]
            ]
            
            if not boxes_scaled:
                continue
            boxes_tensor = torch.tensor(boxes_scaled, device=config.DEVICE, dtype=torch.float32)
            box_indices = torch.zeros(len(boxes_tensor), 1, device=config.DEVICE)
            # 3. RoI Align
            with torch.no_grad():
                roi_feats = roi_align(
                    feat_map, 
                    torch.cat([box_indices, boxes_tensor], dim=1),
                    output_size=(config.ROI_SIZE, config.ROI_SIZE),
                    aligned=True
                )
            self.patient_bank[f"{pid_prefix}.nii"].append(roi_feats.cpu())
                    # 聚合每个病人的特征
        sequences, labels, patient_ids = [], [], []
        for pid, feats_list in self.patient_bank.items():
            if pid in config.ALL_PATIENTS_LABELS:
                all_feats = torch.cat(feats_list, dim=0)
                sequences.append(all_feats)
                labels.append(config.ALL_PATIENTS_LABELS[pid])
                patient_ids.append(pid)
                logging.debug(f"Patient {pid}: {all_feats.shape[0]} total RoIs aggregated.")
        # return sequences, labels, patient_ids
        feat_maps = self.sequences[idx]
        label = self.labels[idx]
  
        num_rois, C, H, W = feat_maps.shape
        
        effective_len = min(num_rois, self.max_rois)
        mask = torch.zeros(self.max_rois, dtype=torch.bool)
        mask[:effective_len] = True

        if num_rois >= self.max_rois:
            padded_maps = feat_maps[:self.max_rois]
        else:
            pad_len = self.max_rois - num_rois
            padding = torch.zeros(pad_len, C, H, W, dtype=feat_maps.dtype)
            padded_maps = torch.cat([feat_maps, padding], dim=0)

        return padded_maps, mask, torch.tensor(label, dtype=torch.long), pid