# classification_pipeline.py

import os
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import roi_align
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import torch.optim as optim
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.ops import roi_align
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from dino_finetune import DINOV2EncoderLoRA
# 从项目根目录导入模块
import config
import models
import utils
import random

# --- 数据集定义 ---
class PatientRoIDataset(Dataset):
    """
    为下游分类器准备数据的数据集。
    负责对每个病人的RoI特征序列进行填充或截断。
    """
    def __init__(self, sequences: List[torch.Tensor], labels: List[int], pids: List[str]):
        self.sequences = sequences
        self.labels = labels
        self.pids = pids
        self.max_rois = config.MAX_ROIS

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        feat_maps = self.sequences[idx]
        label = self.labels[idx]
        pid = self.pids[idx]
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

# --- 特征提取核心函数 ---
# def extract_features_for_pids(
#     pids: List[str],
#     segmentation_model: nn.Module,
#     feature_backbone: nn.Module,
#     fold_num: int,
#     dataset_type: str
# ) -> Tuple[List[torch.Tensor], List[int], List[str]]:
#     """
#     为给定的病人ID列表提取RoI特征。

#     Args:
#         pids (List[str]): 病人ID列表。
#         segmentation_model (nn.Module): 加载了当前折权重的分割模型。
#         feature_backbone (nn.Module): ResNet50特征提取器。
#         fold_num (int): 当前折数，用于缓存路径。
#         dataset_type (str): 'train', 'val', 或 'test'，用于日志和缓存。

#     Returns:
#         A tuple containing (sequences, labels, patient_ids).
#     """
#     # 检查缓存
#     cache_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}", "features")
#     os.makedirs(cache_dir, exist_ok=True)
#     cache_path = os.path.join(cache_dir, f"{dataset_type}_features.pth")

#     if os.path.exists(cache_path):
#         logging.info(f"Loading cached {dataset_type} features from {cache_path}")
#         return torch.load(cache_path)

#     logging.info(f"Extracting RoI features for {len(pids)} patients ({dataset_type} set)...")
    
#     segmentation_model.eval()
#     feature_backbone.eval()

#     # 图像预处理
#     transform = transforms.Compose([
#         transforms.Resize(config.SEG_IMG_DIM, interpolation=transforms.InterpolationMode.BIC_UBIC),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     patient_bank = defaultdict(list)
#     image_files = utils.get_image_files_for_pids(config.IMAGE_DIRS, pids)

#     progress_bar = tqdm(image_files, desc=f"Extracting Features ({dataset_type})")
#     for img_path in progress_bar:
#         pid_prefix = os.path.basename(img_path).split('_')[0]
        
#         image = Image.open(img_path).convert("RGB")
#         tensor = transform(image).unsqueeze(0).to(config.DEVICE)

#         with torch.no_grad():
#             # 1. 分割
#             seg_logits = segmentation_model(tensor)
#             probs = torch.softmax(seg_logits, dim=1)
#             mask = (probs[:, 1] > config.DETECTION_THRESHOLD)[0].cpu().numpy().astype(np.uint8) * 255
            
#             # 2. 提取特征图
#             feat_map = feature_backbone(tensor)

#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             continue

#         spatial_scale_x = feat_map.shape[3] / tensor.shape[3]
#         spatial_scale_y = feat_map.shape[2] / tensor.shape[2]
        
#         boxes_scaled = [
#             [
#                 c[0] * spatial_scale_x, 
#                 c[1] * spatial_scale_y, 
#                 (c[0] + c[2]) * spatial_scale_x, 
#                 (c[1] + c[3]) * spatial_scale_y
#             ]
#             for cnt in contours for c in [cv2.boundingRect(cnt)]
#         ]
        
#         if not boxes_scaled:
#             continue

#         boxes_tensor = torch.tensor(boxes_scaled, device=config.DEVICE, dtype=torch.float32)
#         box_indices = torch.zeros(len(boxes_tensor), 1, device=config.DEVICE)

#         # 3. RoI Align
#         with torch.no_grad():
#             roi_feats = roi_align(
#                 feat_map, 
#                 torch.cat([box_indices, boxes_tensor], dim=1),
#                 output_size=(config.ROI_SIZE, config.ROI_SIZE),
#                 aligned=True
#             )
        
#         patient_bank[f"{pid_prefix}.nii"].append(roi_feats.cpu())

#     # 聚合每个病人的特征
#     sequences, labels, patient_ids = [], [], []
#     for pid, feats_list in patient_bank.items():
#         if pid in config.ALL_PATIENTS_LABELS:
#             all_feats = torch.cat(feats_list, dim=0)
#             sequences.append(all_feats)
#             labels.append(config.ALL_PATIENTS_LABELS[pid])
#             patient_ids.append(pid)
#             logging.debug(f"Patient {pid}: {all_feats.shape[0]} total RoIs aggregated.")

#     # 保存到缓存
#     torch.save((sequences, labels, patient_ids), cache_path)
#     logging.info(f"Saved extracted {dataset_type} features to {cache_path}")
    
#     return sequences, labels, patient_ids
def extract_features_for_pids(
    pids: List[str],
    segmentation_model: nn.Module,
    feature_backbone: nn.Module,
    fold_num: int,
    dataset_type: str,
    class_aug_ratio: Dict[int, int],   # 新增：{label: K}
) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """
    在线分割获得掩码（回投到原图），再对(原图, 掩码)做同步增广；提取 RoI 特征并按“增广版本”聚合。
    返回 (sequences, labels, patient_ids)；train 集会生成 pid_aug{k} 的多条样本。
    """

    # ---------------------- 缓存 ----------------------
    cache_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}", "features_online_aug")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_type}_features.pth")
    if os.path.exists(cache_path):
        logging.info(f"Loading cached {dataset_type} features from {cache_path}")
        return torch.load(cache_path)

    logging.info(f"Extracting RoI features for {len(pids)} patients ({dataset_type})...")

    device = getattr(config, "DEVICE", "cuda")
    thr = float(getattr(config, "DETECTION_THRESHOLD", 0.5))
    MIN_AREA = getattr(config, "MIN_ROI_AREA", 0)  # 0 表示不启用

    segmentation_model.eval()
    feature_backbone.eval()

    # 1) 分割输入：确定性预处理（仅供分割）
    seg_tf = A.Compose([
        A.LongestMaxSize(max_size=392),
        A.PadIfNeeded(min_height=392, min_width=392, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 2) 增广：作用于(原图, 回投掩码)；几何同步、像素只对图像
    if dataset_type == 'train':
        slice_tf = A.ReplayCompose([
            A.Rotate(limit=30, p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomScale(scale_limit=0.2, p=0.2),
            A.LongestMaxSize(max_size=392),
            A.PadIfNeeded(392, 392, border_mode=cv2.BORDER_CONSTANT, value=0),
            # 像素增强在 Normalize 之前
            A.RandomGamma(gamma_limit=(70, 150), p=0.1),
            # A.GaussNoise(var_limit=(0.0, 10.0), p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else :
        slice_tf = A.ReplayCompose([
            A.LongestMaxSize(max_size=392),
            A.PadIfNeeded(392, 392, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),])


    # 3) 按 pid 分组切片路径（需要你已有的工具）
    #    如果没有 utils.group_slice_paths_by_pid，请改成你自己的实现
    pid2imgs = utils.group_slice_paths_by_pid(config.IMAGE_DIRS, pids)

    sequences: List[torch.Tensor] = []
    labels: List[int] = []
    patient_ids: List[str] = []

    for pid in tqdm(pids, desc=f"Patients ({dataset_type})"):
        img_paths = sorted(
            pid2imgs.get(pid, []),
            key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
        )
        if not img_paths:
            logging.warning(f"No slices found for {pid}, skipped.")
            continue

        # ---- A) 在线推理整套掩码，并回投到原图坐标系（0/255） ----
        back_masks: List[np.ndarray] = []
        orig_images: List[np.ndarray] = []

        for p in img_paths:
            img_rgb = np.array(Image.open(p).convert("RGB"))
            H0, W0 = img_rgb.shape[:2]
            orig_images.append(img_rgb)

            seg_in = seg_tf(image=img_rgb)['image'].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = segmentation_model(seg_in)            # (1,C,H',W')
                prob   = torch.softmax(logits, dim=1)[0, 1]    # (H',W')
                prob_np = prob.float().cpu().numpy()

            # 回投到原图大小
            prob_back = cv2.resize(prob_np, (W0, H0), interpolation=cv2.INTER_LINEAR)
            mask_255  = (prob_back > thr).astype(np.uint8) * 255
            back_masks.append(mask_255)

        # ---- B) 计算该病人的增广次数 K ----
        label = config.ALL_PATIENTS_LABELS[pid]
        K = class_aug_ratio.get(label, 1) if dataset_type == 'train' else 1
        base_seed = (hash(pid) & 0x7FFFFFFF)

        # ---- C) 做 K 次增广；同一次增广内各切片共享同一组参数（Replay） ----
        for k in range(K):
            aug_seed = base_seed + k
            random.seed(aug_seed); np.random.seed(aug_seed); 

            roi_list = []
            replay = None

            for img_np, m_np in zip(orig_images, back_masks):
                if replay is None:
                    out = slice_tf(image=img_np, mask=m_np)
                    replay = out['replay']
                else:
                    out = A.ReplayCompose.replay(replay, image=img_np, mask=m_np)

                img_t  = out['image'].unsqueeze(0).to(device)   # (1,3,392,392)
                mask_t = out['mask'].cpu().numpy().squeeze().astype(np.uint8)                      # (392,392) uint8 0/255

                with torch.no_grad():
                    feat_map = feature_backbone(img_t)          # (1,C,Hf,Wf)

                contours, _ = cv2.findContours(mask_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                Hf, Wf = feat_map.shape[2], feat_map.shape[3]
                Hi, Wi = img_t.shape[2], img_t.shape[3]
                sy, sx = Hf / Hi, Wf / Wi

                rects = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if MIN_AREA and (w * h) < MIN_AREA:
                        continue
                    x1, y1 = x * sx, y * sy
                    x2, y2 = (x + w) * sx, (y + h) * sy
                    # clamp
                    x1 = max(0.0, min(float(Wf - 1), x1))
                    y1 = max(0.0, min(float(Hf - 1), y1))
                    x2 = max(0.0, min(float(Wf),     x2))
                    y2 = max(0.0, min(float(Hf),     y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    rects.append([x1, y1, x2, y2])

                if not rects:
                    continue

                boxes_t = torch.tensor(rects, dtype=torch.float32, device=device)
                batch_idx = torch.zeros(len(boxes_t), 1, dtype=torch.float32, device=device)

                with torch.no_grad():
                    roi_feats = roi_align(
                        feat_map,
                        torch.cat([batch_idx, boxes_t], dim=1),
                        output_size=(config.ROI_SIZE, config.ROI_SIZE),
                        aligned=True
                    )
                roi_list.append(roi_feats.cpu())

            # 本次增广（k）结束：聚合成一条 sample
            if roi_list:
                feats = torch.cat(roi_list, dim=0)  # (#RoI, C, H, W)
                sequences.append(feats)
                labels.append(label)
                patient_ids.append(f"{pid}_aug{k}" if dataset_type == 'train' else pid)

    # ---------------------- 缓存与返回 ----------------------
    torch.save((sequences, labels, patient_ids), cache_path)
    logging.info(f"Saved extracted {dataset_type} features to {cache_path}")
    return sequences, labels, patient_ids

# --- 训练与评估函数 ---
def evaluate_classifier(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module,
    device: str,
    is_test_set: bool = False
) -> Dict[str, Any]:
    """
    在验证集或测试集上评估分类器。
    """
    model.eval()
    total_loss = 0.0
    all_targets, all_pred_probs = [], []

    with torch.no_grad():
        for feat_maps, roi_masks, targets, _ in loader:
            feat_maps, roi_masks, targets = feat_maps.to(device), roi_masks.to(device), targets.to(device)
            
            logits = model(feat_maps, roi_masks)
            loss = criterion(logits, targets)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_targets.extend(targets.cpu().numpy())
            all_pred_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    
    # 计算指标
    metrics = {'loss': avg_loss}
    # if len(np.unique(all_targets)) > 1:
    #     metrics['auc'] = roc_auc_score(all_targets, all_pred_probs)
    #     fpr, tpr, thresholds = roc_curve(all_targets, all_pred_probs)
    #     # 找到最佳阈值 (Youden's J statistic)
    #     optimal_idx = np.argmax(tpr - fpr)
    #     optimal_threshold = thresholds[optimal_idx]
    # else: # 只有一个类别
    #     metrics['auc'] = 0.5
    #     optimal_threshold = 0.5
    if len(np.unique(all_targets)) > 1:
        metrics['auc'] = roc_auc_score(all_targets, all_pred_probs)
        
        # 使用F1-score最大来选择最优阈值
        f1_scores = []
        thresholds = np.linspace(0, 1, 5001)  # 生成0到1之间的101个阈值
        
        for threshold in thresholds:
            binary_preds = (np.array(all_pred_probs) >= threshold).astype(int)
            f1 = f1_score(all_targets, binary_preds, zero_division=0)
            f1_scores.append(f1)
        
        # 找到F1-score最大的阈值
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        max_f1 = f1_scores[optimal_idx]
        
    else:
        metrics['auc'] = 0.5
        optimal_threshold = 0.5
        max_f1 = 0.0
    metrics['optimal_threshold'] = optimal_threshold
    
    # 使用最佳阈值计算其他指标
    binary_preds = (np.array(all_pred_probs) >= optimal_threshold).astype(int)
    metrics['accuracy'] = accuracy_score(all_targets, binary_preds)
    metrics['sensitivity'] = recall_score(all_targets, binary_preds, zero_division=0)
    metrics['precision'] = precision_score(all_targets, binary_preds, zero_division=0)
    metrics['f1_score'] = f1_score(all_targets, binary_preds, zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    except ValueError:
        metrics['specificity'] = 0

    return metrics


def run_classification_training(
    fold_num: int, 
    train_pids: List[str], 
    val_pids: List[str], 
    segmentation_model_path: str
) -> Tuple[str, float]:
    """
    为指定的折执行完整的分类模型训练流程。
    """
    logging.info(f"========== Starting Classification Training for Fold {fold_num} ==========")
    
    # 1. 加载当前折的分割模型和特征提取器
    logging.info("Loading models for feature extraction...")
    from dinov2_loader import load_dinov2_base_encoder

    base_encoder = load_dinov2_base_encoder(
        config.DINOV2_REPO_PATH, config.DINOV2_PRETRAINED_WEIGHTS
    )
    segmentation_model = DINOV2EncoderLoRA(
        encoder=base_encoder, r=config.SEG_R_LORA, emb_dim=config.SEG_EMB_DIM, img_dim=config.SEG_IMG_DIM,
        n_classes=config.SEG_N_CLASSES, use_lora=config.USE_LORA, use_fpn=True
    ).to(config.DEVICE)
    segmentation_model.load_parameters(segmentation_model_path)
    
    feature_backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-4]).to(config.DEVICE)

    # 2. 提取训练集和验证集的特征
    train_seqs, train_y, train_pids_out = extract_features_for_pids(train_pids, segmentation_model, feature_backbone, fold_num, 'train',class_aug_ratio= {0: 5,   1: 5})
    val_seqs, val_y, val_pids_out = extract_features_for_pids(val_pids, segmentation_model, feature_backbone, fold_num, 'val',class_aug_ratio={0: 1, 1: 1})

    train_ds = PatientRoIDataset(train_seqs, train_y, train_pids_out)
    val_ds = PatientRoIDataset(val_seqs, val_y, val_pids_out)
    
    train_loader = DataLoader(train_ds, batch_size=config.CLS_BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.CLS_BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 3. 初始化分类模型
    logging.info("Initializing Transformer classifier model...")
    pos_embedder = models.PatchPositionEmbedding(grid_size=config.ROI_SIZE, dim=config.BACKBONE_OUTPUT_DIM, learnable=config.CLS_POS_EMBED_LEARNABLE)
    transformer = models.ROIBasedTransformerClassifier(
        input_dim=config.BACKBONE_OUTPUT_DIM, hidden_dim=config.CLS_HIDDEN_DIM,
        n_heads=config.CLS_N_HEADS, n_layers=config.CLS_N_LAYERS
    )
    model = models.FullTumorClassifier(pos_embedder, transformer).to(config.DEVICE)

    # 4. 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.CLS_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)

    # 5. 训练循环
    best_val_auc = -1.0
    best_model_info = {}
    fold_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}")
    best_model_path = os.path.join(fold_dir, "best_classifier_model.pth")
    
    logging.info(f"Starting training for {config.CLS_EPOCHS} epochs...")
    for epoch in range(config.CLS_EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.CLS_EPOCHS} [Training]")
        for feat_maps, roi_masks, targets, _ in progress_bar:
            feat_maps, roi_masks, targets = feat_maps.to(config.DEVICE), roi_masks.to(config.DEVICE), targets.to(config.DEVICE)
            
            optimizer.zero_grad()
            logits = model(feat_maps, roi_masks)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        val_metrics = evaluate_classifier(model, val_loader, criterion, config.DEVICE)
        
        logging.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val AUC: {val_metrics['auc']:.4f}"
        )
        
        scheduler.step(val_metrics['auc'])

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            logging.info(f"🎉 New best validation AUC: {best_val_auc:.4f}. Saving model...")
            deep_copied_state_dict = {}
            for key, value in model.state_dict().items():
                    deep_copied_state_dict[key] = value.clone().detach()
            best_model_info = {
                'model_state_dict': deep_copied_state_dict,
                'optimal_threshold': val_metrics['optimal_threshold'],
                'val_auc': best_val_auc
            }
            torch.save(best_model_info, best_model_path)
            
    logging.info(f"========== Classification Training for Fold {fold_num} Finished ==========")
    return best_model_path


def run_classification_testing(
    fold_num: int, 
    test_pids: List[str],
    segmentation_model_path: str,
    classifier_model_path: str
) -> Dict[str, Any]:
    """
    在测试集上评估最终的分类模型。
    """
    logging.info(f"========== Starting Classification Testing for Fold {fold_num} ==========")
    
    # 1. 加载模型
    logging.info("Loading models for testing...")
    from dinov2_loader import load_dinov2_base_encoder

    base_encoder = load_dinov2_base_encoder(
        config.DINOV2_REPO_PATH, config.DINOV2_PRETRAINED_WEIGHTS
    )
    segmentation_model = DINOV2EncoderLoRA(
        encoder=base_encoder, r=config.SEG_R_LORA, emb_dim=config.SEG_EMB_DIM, img_dim=config.SEG_IMG_DIM,
        n_classes=config.SEG_N_CLASSES, use_lora=config.USE_LORA, use_fpn=True
    ).to(config.DEVICE)
    segmentation_model.load_parameters(segmentation_model_path)
    
    feature_backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-4]).to(config.DEVICE)

    # 2. 提取测试集特征
    test_seqs, test_y, test_pids_out = extract_features_for_pids(test_pids, segmentation_model, feature_backbone, fold_num, 'test',class_aug_ratio={0: 1, 1: 1})
    test_ds = PatientRoIDataset(test_seqs, test_y, test_pids_out)
    test_loader = DataLoader(test_ds, batch_size=config.CLS_BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. 初始化分类模型并加载最佳权重
    pos_embedder = models.PatchPositionEmbedding(grid_size=config.ROI_SIZE, dim=config.BACKBONE_OUTPUT_DIM, learnable=config.CLS_POS_EMBED_LEARNABLE)
    transformer = models.ROIBasedTransformerClassifier(
        input_dim=config.BACKBONE_OUTPUT_DIM, hidden_dim=config.CLS_HIDDEN_DIM,
        n_heads=config.CLS_N_HEADS, n_layers=config.CLS_N_LAYERS
    )
    model = models.FullTumorClassifier(pos_embedder, transformer).to(config.DEVICE)
    
    checkpoint = torch.load(classifier_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimal_threshold_from_val = checkpoint['optimal_threshold']
    
    logging.info(f"Loaded best classifier model. Using threshold from validation: {optimal_threshold_from_val:.4f}")

    # 4. 在测试集上评估
    model.eval()
    all_targets, all_pred_probs = [], []
    all_pids = []
    with torch.no_grad():
        for feat_maps, roi_masks, targets, pids_batch in tqdm(test_loader, desc="Testing"):
            feat_maps, roi_masks = feat_maps.to(config.DEVICE), roi_masks.to(config.DEVICE)
            logits = model(feat_maps, roi_masks)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_targets.extend(targets.cpu().numpy())
            all_pred_probs.extend(probs.cpu().numpy())
            all_pids.extend(pids_batch)

    # 使用从验证集得到的固定阈值进行评估
    binary_preds = (np.array(all_pred_probs) >= optimal_threshold_from_val).astype(int)
    
    test_metrics = {}
    if len(np.unique(all_targets)) > 1:
        test_metrics['auc'] = roc_auc_score(all_targets, all_pred_probs)
    else:
        test_metrics['auc'] = 0.5
        
    test_metrics['accuracy'] = accuracy_score(all_targets, binary_preds)
    test_metrics['sensitivity'] = recall_score(all_targets, binary_preds, zero_division=0)
    test_metrics['precision'] = precision_score(all_targets, binary_preds, zero_division=0)
    test_metrics['f1_score'] = f1_score(all_targets, binary_preds, zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds).ravel()
        test_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    except ValueError:
        test_metrics['specificity'] = 0

    logging.info("--- Test Set Evaluation Report ---")
    for key, value in test_metrics.items():
        logging.info(f"{key.capitalize():<15}: {value:.4f}")
    for i, (pred, target) in enumerate(zip(binary_preds, all_targets)):
        if pred != target:
            logging.info(f"  - PID: {all_pids[i]}, True: {target}, Predicted: {pred}, Probability_for_1: {all_pred_probs[i]:.4f}")

    logging.info(f"Confusion Matrix:\n{confusion_matrix(all_targets, binary_preds)}")
    logging.info(f"========== Classification Testing for Fold {fold_num} Finished ==========")
    
    return test_metrics