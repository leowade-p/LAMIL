import os
import logging
from typing import List, Dict, Any, Tuple
from typing import Iterator
from collections import defaultdict
from dino_finetune import DINOV2EncoderLoRA
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import roi_align
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from scipy.ndimage import label
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import config
import models
import utils
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models_mil import MILClassifier
from models_method_abmil import ABMILClassifier
from models_method_dsmil import DSMILClassifier
from models_method_transmil import TransMILClassifier
from models_method_dtfd import DTFDClassifier
from models_method_clam import CLAMSBClassifier
from models_method_camil import CAMILClassifier
from models_method_rrtmil import RRTMILClassifier
# =================================================================================
# Section 1: 全新的Dataset和Collate Function
# =================================================================================

class PatientComponentDataset(Dataset):
    """
    【新】这个Dataset非常简单，仅负责根据索引返回一个病人的完整数据。
    数据结构为: {'pid': str, 'label': int, 'sequences': List[torch.Tensor]}
    其中 'sequences' 列表包含了该病人所有连通分量的RoI特征序列。
    """
    def __init__(self, patient_data: List[Dict[str, Any]]):
        self.patient_data = patient_data

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, idx):
        return self.patient_data[idx]

def compute_mil_loss(instance_logits, labels, patient_indices, w_plus=0.01, w_minus=1.0):
    """
    计算多实例学习的不对称损失
    
    Args:
        instance_logits: 实例级别的logits [N_instances, 2]
        labels: 包级别标签 [N_patients]
        patient_indices: 实例到病人的映射 [N_instances]
        w_plus: 正样本损失权重
        w_minus: 负样本损失权重
    
    Returns:
        total_loss: 总损失
        loss_positive: 正样本损失
        loss_negative: 负样本损失
    """
    loss_positive = torch.tensor(0.0, device=instance_logits.device)
    loss_negative = torch.tensor(0.0, device=instance_logits.device)
    
    if instance_logits is None or instance_logits.shape[0] == 0:
        return loss_positive + loss_negative, loss_positive, loss_negative
    # logging.info(f'instance_logits{instance_logits.shape}')
    instance_probs_pos = F.softmax(instance_logits, dim=1)[:, 1]
    
    # 分别处理恶性病例和良性病例
    positive_mask = (labels == 1)
    negative_mask = (labels == 0)
    
    # --- 处理恶性病例 (Positive Bags) ---
    if torch.any(positive_mask):
        positive_bag_indices = positive_mask.nonzero().flatten()
        max_instance_probs = []
        
        for i in positive_bag_indices:
            mask_i = (patient_indices == i)
            if torch.any(mask_i):
                max_prob, _ = torch.max(instance_probs_pos[mask_i], dim=0)
                max_instance_probs.append(max_prob)
        
        if max_instance_probs:
            max_instance_probs_tensor = torch.stack(max_instance_probs)
            positive_labels = torch.ones_like(max_instance_probs_tensor)
            loss_positive = F.binary_cross_entropy(max_instance_probs_tensor, positive_labels, reduction='sum')
    
    # --- 处理良性病例 (Negative Bags) ---
    if torch.any(negative_mask):
        negative_instance_mask = (labels[patient_indices] == 0)
        
        if torch.any(negative_instance_mask):
            negative_instance_probs = instance_probs_pos[negative_instance_mask]
            negative_labels = torch.zeros_like(negative_instance_probs)
            loss_negative = F.binary_cross_entropy(negative_instance_probs, negative_labels, reduction='sum')
    
    total_loss = w_plus * loss_positive + w_minus * loss_negative
    return total_loss, loss_positive, loss_negative


def compute_roi_loss(bag_logits, instance_logits, labels, 
                    component_patient_indices, instance_patient_indices, instance_component_indices,
                    criterion_sum):
    """
    计算ROI级别的损失 (L2损失)
    
    Args:
        bag_logits: 包级别logits [N_components, 2]
        instance_logits: 实例级别logits [N_instances, 2]
        labels: 包级别标签 [N_patients]
        component_patient_indices: 组件到病人的映射 [N_components]
        instance_patient_indices: 实例到病人的映射 [N_instances]
        instance_component_indices: 实例到组件的映射 [N_instances]
        criterion_sum: 损失函数
    
    Returns: 
        batch_l1_minus: 良性病人的组件级别损失列表
        batch_l2_minus: 良性病人的实例级别损失列表
    """
    batch_l2_plus,  batch_l2_minus = [], []
    
    if bag_logits is None or instance_logits is None:
        return  batch_l2_plus, batch_l2_minus
    
    num_patients = len(labels)
    
    for i in range(num_patients):
        patient_label = labels[i]
        is_patient_i_component_mask = (component_patient_indices == i)
        is_patient_i_instance_mask = (instance_patient_indices == i)
        
        if not torch.any(is_patient_i_component_mask): 
            continue

        if patient_label == 1:  # --- 处理恶性病人 ---
            # --- 计算 L1+ (组件级别损失) ---
            patient_component_logits = bag_logits[is_patient_i_component_mask]
            if patient_component_logits.shape[0] > 0:
                component_scores = patient_component_logits[:, 1]
                _, champion_local_idx = torch.max(component_scores, dim=0)
            # --- 计算 L2+ (实例级别损失) ---
            patient_component_global_indices = is_patient_i_component_mask.nonzero().flatten()
            if champion_local_idx < len(patient_component_global_indices):
                champion_component_global_id = patient_component_global_indices[champion_local_idx]
                is_champion_component_mask = (instance_component_indices == champion_component_global_id)
                champion_instance_logits = instance_logits[is_champion_component_mask]
                
                if champion_instance_logits.shape[0] > 0:
                    champion_labels = torch.ones(champion_instance_logits.shape[0], 
                                               dtype=torch.long, device=instance_logits.device)
                    l2_plus = criterion_sum(champion_instance_logits, champion_labels)
                    batch_l2_plus.append(l2_plus)

        else:  # --- 处理良性病人 ---
            # --- 计算 L1- (组件级别损失) ---
            patient_component_logits = bag_logits[is_patient_i_component_mask]
            if patient_component_logits.shape[0] > 0:
                negative_labels = torch.zeros(patient_component_logits.shape[0], 
                                            dtype=torch.long, device=bag_logits.device)
            # --- 计算 L2- (实例级别损失) ---
            patient_instance_logits = instance_logits[is_patient_i_instance_mask]
            if patient_instance_logits.shape[0] > 0:
                negative_labels = torch.zeros(patient_instance_logits.shape[0], 
                                            dtype=torch.long, device=instance_logits.device)
                l2_minus = criterion_sum(patient_instance_logits, negative_labels)
                batch_l2_minus.append(l2_minus)
    
    return batch_l2_plus,  batch_l2_minus


def patient_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    【双轨并行版】自定义collate_fn。
    除了打包序列和掩码，还创建并返回三个层次的索引。
    """
    # --- 连通分量级 ---
    all_sequences = []
    all_masks = []
    component_patient_indices = []
    
    # --- 实例 (RoI) 级 ---
    instance_patient_indices = []
    instance_component_indices = []
    
    # --- 病人级 ---
    labels = []
    pids = []
    
    component_global_idx = 0 # 用于追踪连通分量的全局唯一ID

    for i, patient_sample in enumerate(batch):
        labels.append(patient_sample['label'])
        pids.append(patient_sample['pid'])
        pid = patient_sample['pid']
        # logging.info(f'pid {pid}')
        if not patient_sample['sequences']:
            continue
        
        # 遍历这个病人的每个连通分量序列
        for seq in patient_sample['sequences']:
            # seq shape: (num_rois, C)
            seq = seq.squeeze() # 确保没有多余维度
            if seq.ndim == 1: seq = seq.unsqueeze(0) # 处理只有一个RoI的情况
            if seq.shape[0] == 0:
                continue # 如果没有RoI，直接跳过这个连通分量
            num_rois, C = seq.shape
            # logging.info(f'num_rois{num_rois}')
            # --- 1. 填充序列和掩码 (逻辑不变) ---
            effective_len = min(num_rois, config.MAX_ROIS)
            if num_rois >= config.MAX_ROIS:
                padded_seq = seq[:config.MAX_ROIS]
                actual_num_rois = config.MAX_ROIS
            else:
                padding = torch.zeros(config.MAX_ROIS - num_rois, C, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding], dim=0)
                actual_num_rois = num_rois
            
            mask = torch.zeros(config.MAX_ROIS, dtype=torch.bool)
            mask[:effective_len] = True

            # --- 2. 收集连通分量级信息 ---
            all_sequences.append(padded_seq)
            all_masks.append(mask)
            component_patient_indices.append(i) # 这个连通分量属于病人 i
            
            # --- 3. 【核心新增】收集实例级索引 ---
            # 为这个连通分量的 num_rois 个有效RoI，记录它们的归属
            instance_patient_indices.extend([i] * actual_num_rois)
            instance_component_indices.extend([component_global_idx] * actual_num_rois)
            
            component_global_idx += 1

    # --- 4. 最终堆叠并返回 ---
    batch_dict = {
        'labels': torch.tensor(labels, dtype=torch.long),
        'pids': pids,
        'num_patients': len(batch)
    }

    if all_sequences:
        batch_dict.update({
            'sequences': torch.stack(all_sequences, dim=0),
            'masks': torch.stack(all_masks, dim=0),
            'component_patient_indices': torch.tensor(component_patient_indices, dtype=torch.long),
            'instance_patient_indices': torch.tensor(instance_patient_indices, dtype=torch.long),
            'instance_component_indices': torch.tensor(instance_component_indices, dtype=torch.long)
        })
    else: # 处理空批次
        batch_dict.update({
            'sequences': torch.empty(0, config.MAX_ROIS, config.BACKBONE_OUTPUT_DIM),
            'masks': torch.empty(0, config.MAX_ROIS, dtype=torch.bool),
            'component_patient_indices': torch.empty(0, dtype=torch.long),
            'instance_patient_indices': torch.empty(0, dtype=torch.long),
            'instance_component_indices': torch.empty(0, dtype=torch.long)
        })

    return batch_dict

# =================================================================================
# Section 2: 重构的特征提取函数
# =================================================================================
def extract_features_for_pids(
    pids: List[str],
    segmentation_model: nn.Module,
    feature_backbone: nn.Module,
    fold_num: int,
    dataset_type: str,
    class_aug_ratio: Dict[int, int]
) -> List[Dict[str, Any]]:
    """
    【重构】为给定的病人ID列表提取RoI特征，并按3D连通分量进行组织。
    """
    cache_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}", "component_features")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_type}_features.pth")

    if os.path.exists(cache_path):
        logging.info(f"Loading cached {dataset_type} features from {cache_path}")
        return torch.load(cache_path)

    logging.info(f"Extracting Component RoI features for {len(pids)} patients ({dataset_type})...")

    device = config.DEVICE
    thr = config.DETECTION_THRESHOLD
    segmentation_model.eval()
    feature_backbone.eval()

    # --- 1. 定义数据增强 ---
    seg_tf = A.Compose([
        A.LongestMaxSize(max_size=392),
        A.PadIfNeeded(min_height=392, min_width=392, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    if dataset_type == 'train':
        slice_tf = A.ReplayCompose([
            A.Rotate(limit=30, p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.RandomScale(scale_limit=0.2, p=0.2),
            A.LongestMaxSize(max_size=392),
            A.PadIfNeeded(392, 392, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.RandomGamma(gamma_limit=(70, 150), p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        slice_tf = A.ReplayCompose([
            A.LongestMaxSize(max_size=392),
            A.PadIfNeeded(392, 392, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    pid2imgs = utils.group_slice_paths_by_pid(config.IMAGE_DIRS, pids)
    final_patient_data = []
   
    for pid in tqdm(pids, desc=f"Patients ({dataset_type})"):
        # --- 2. 预处理：获取病人所有原始图像和掩码 ---
        img_paths = sorted(
            pid2imgs.get(pid, []),
            key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
        )
        logging.info(pid)
        logging.info(len(img_paths))
        if not img_paths: continue

        orig_images, back_masks = [], []
        for p in img_paths:
            img_rgb = np.array(Image.open(p).convert("RGB"))
            H0, W0 = img_rgb.shape[:2]
            # logging.info(p)
            # logging.info(W0)
            # logging.info(H0)
            img_resized = cv2.resize(img_rgb, (392, 392), interpolation=cv2.INTER_LINEAR)
            orig_images.append(img_resized)

            # 使用已经统一尺寸的图像进行分割
            seg_in = seg_tf(image=img_resized)['image'].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = segmentation_model(seg_in)
                prob = torch.softmax(logits, dim=1)[0, 1]
                
                # **关键改动**：不再缩放回原始尺寸，直接使用模型输出尺寸的掩码
                mask_392 = (prob > thr).cpu().numpy().astype(np.uint8)
                back_masks.append(mask_392)

        if not back_masks: continue

        # --- 3. 核心：在数据增强前进行3D连通域分析 ---
        try:
            binary_3d_mask = np.stack(back_masks, axis=0)
        except ValueError:
            logging.warning(f"Inconsistent mask shapes for {pid}, skipping.")
            continue
        
        s = np.zeros((3, 3, 3), dtype=bool); s[1, 1, :] = True; s[1, :, 1] = True; s[:, 1, 1] = True
        labeled_array, num_components = label(binary_3d_mask, structure=s)
        # logging.info(labeled_array)
        # logging.info(num_components)
        if num_components == 0: continue

        label_val = config.ALL_PATIENTS_LABELS[pid]
        K = class_aug_ratio.get(label_val, 1) if dataset_type == 'train' else 1
        base_seed = (hash(pid) & 0x7FFFFFFF)

        # --- 4. 核心：带ID图的数据增强与特征提取 ---
        for k in range(K):
            aug_seed = base_seed + k
            random.seed(aug_seed); np.random.seed(aug_seed)
            
            component_feats_this_aug = defaultdict(list)
            replay = None

            for i, (img_np, m_np) in enumerate(zip(orig_images, back_masks)):
                id_map_slice = labeled_array[i] # 获取对应的2D连通分量ID图
                
                if replay is None:
                    # **关键**：同时增强图像、掩码和ID图
                    # 为ID图指定最近邻插值，防止ID值被破坏
                    out = slice_tf(image=img_np, masks=[m_np, id_map_slice.astype(np.int32)])
                    replay = out['replay']
                else:
                    out = A.ReplayCompose.replay(replay, image=img_np, masks=[m_np, id_map_slice.astype(np.int32)])
                
                img_t = out['image'].unsqueeze(0).to(device)
                aug_id_map = out['masks'][1].cpu().numpy() # 增强后的ID图

                unique_ids = np.unique(aug_id_map)
                unique_ids = unique_ids[unique_ids != 0]
                if len(unique_ids) == 0: continue

                with torch.no_grad():
                    feat_map = feature_backbone(img_t)

                Hf, Wf = feat_map.shape[2], feat_map.shape[3]
                Hi, Wi = img_t.shape[2], img_t.shape[3]
                sy, sx = Hf / Hi, Wf / Wi

                for comp_id in unique_ids:
                    # 为每个分量ID创建临时掩码并提取其包围框
                    comp_mask = (aug_id_map == comp_id).astype(np.uint8)
                    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours: continue
                    
                    # 通常一个分量在一个切片上是一个连通区域，但增强后可能碎裂
                    # 这里我们将它们合并处理，取所有碎块的最大外接矩形
                    all_points = np.concatenate(contours, axis=0)
                    x, y, w, h = cv2.boundingRect(all_points)
                    
                    box = torch.tensor([[x * sx, y * sy, (x + w) * sx, (y + h) * sy]], device=device)
                    
                    with torch.no_grad():
                        roi_feat = roi_align(feat_map, torch.cat([torch.zeros(1, 1, device=device), box], dim=1),
                                             output_size=(1, 1), aligned=True)
                    component_feats_this_aug[comp_id].append(roi_feat.cpu())
            
            # --- 5. 聚合本次增强的结果 ---
            if not component_feats_this_aug: continue

            sequences_for_patient = []
            for comp_id in sorted(component_feats_this_aug.keys()):
                feats = torch.cat(component_feats_this_aug[comp_id], dim=0)
                sequences_for_patient.append(feats)
            
            final_patient_data.append({
                'pid': f"{pid}_aug{k}" if dataset_type == 'train' else pid,
                'label': label_val,
                'sequences': sequences_for_patient
            })

    torch.save(final_patient_data, cache_path)
    logging.info(f"Saved extracted {dataset_type} features to {cache_path}")
    return final_patient_data

# In modules/classification_pipeline.py

def run_classification_training(
    fold_num: int, 
    train_pids: List[str], 
    val_pids: List[str], 
    segmentation_model_path: str,
    lr,
    roi,
    w_plus

) -> Tuple[str, float]:
    logging.info(f"========== Starting 'Dual-Track' MIL Training for Fold {fold_num} ==========")
    
    # --- 1. 初始化 (模型加载，数据提取，Dataloader创建) ---
    # (这部分代码保持不变)
    training_history = {
        'train_loss': [], 'val_loss': [], 'val_auc': [],
        'loss_l1_plus': [], 'loss_l2_plus': [], 'loss_l1_minus': [], 'loss_l2_minus': [],
        "loss_positive":[], "loss_negative":[]
    }
    from dinov2_loader import load_dinov2_base_encoder

    base_encoder = load_dinov2_base_encoder(
        config.DINOV2_REPO_PATH, config.DINOV2_PRETRAINED_WEIGHTS
    )
    segmentation_model = DINOV2EncoderLoRA(
        encoder=base_encoder, r=config.SEG_R_LORA, emb_dim=config.SEG_EMB_DIM, img_dim=config.SEG_IMG_DIM,
        n_classes=config.SEG_N_CLASSES, use_lora=True, use_fpn=False
    ).to(config.DEVICE)
    segmentation_model.load_parameters(segmentation_model_path)
    feature_backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-4]).to(config.DEVICE)
    train_data = extract_features_for_pids(train_pids, segmentation_model, feature_backbone, fold_num, 'train', class_aug_ratio=config.CLASS_AUG_RATIO)
    val_data = extract_features_for_pids(val_pids, segmentation_model, feature_backbone, fold_num, 'val', class_aug_ratio={0: 1, 1: 1})
    train_ds = PatientComponentDataset(train_data)
    val_ds = PatientComponentDataset(val_data)
    train_loader = DataLoader(train_ds, batch_size=config.CLS_BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=patient_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.CLS_BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=patient_collate_fn)
    
    # --- 2. 初始化“双轨并行”的 MILClassifier 模型 ---
    logging.info("Initializing 'Dual-Track' MIL classifier model...")
    model_type = getattr(config, "CLS_MODEL_TYPE", "mil_asym").lower()
    if model_type == "abmil":
        model = ABMILClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS,
            attention_dim=getattr(config, "ABMIL_ATTENTION_DIM", 128),
            gated=bool(getattr(config, "ABMIL_GATED", True)),
        ).to(config.DEVICE)
        logging.info("Using ABMIL classifier.")
    elif model_type == "dsmil":
        model = DSMILClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS,
            q_dim=int(getattr(config, "DSMIL_Q_DIM", 128)),
            nonlinear=bool(getattr(config, "DSMIL_NONLINEAR", True)),
            passing_v=bool(getattr(config, "DSMIL_PASSING_V", False)),
            dropout_v=float(getattr(config, "DSMIL_DROPOUT_V", 0.0)),
        ).to(config.DEVICE)
        logging.info("Using DSMIL classifier.")
    elif model_type == "transmil":
        model = TransMILClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS,
        ).to(config.DEVICE)
        logging.info("Using TransMIL classifier.")
    elif model_type == "dtfd":
        model = DTFDClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS,
            n_classes=2,
            num_group=int(getattr(config, "DTFD_NUM_GROUP", 4)),
            total_instance=int(getattr(config, "DTFD_TOTAL_INSTANCE", 4)),
            distill_type=str(getattr(config, "DTFD_DISTILL_TYPE", "AFS")),
            num_res_layers=int(getattr(config, "DTFD_NUM_RES_LAYERS", 0)),
            droprate_tier1=float(getattr(config, "DTFD_DROPRATE_TIER1", 0.0)),
            droprate_tier2=float(getattr(config, "DTFD_DROPRATE_TIER2", 0.0)),
            attn_dim=int(getattr(config, "DTFD_ATTN_DIM", 128)),
        ).to(config.DEVICE)
        logging.info("Using DTFD-MIL classifier.")
    elif model_type == "clam":
        model = CLAMSBClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS,
            n_classes=2,
            gate=bool(getattr(config, "CLAM_GATE", True)),
            attn_hidden_dim=int(getattr(config, "CLAM_ATTN_HIDDEN_DIM", 256)),
            dropout=float(getattr(config, "CLAM_DROPOUT", 0.25)),
            k_sample=int(getattr(config, "CLAM_K_SAMPLE", 8)),
            subtyping=bool(getattr(config, "CLAM_SUBTYPING", False)),
            no_inst_cluster=bool(getattr(config, "CLAM_NO_INST_CLUSTER", False)),
            inst_loss_type=str(getattr(config, "CLAM_INST_LOSS", "ce")),
        ).to(config.DEVICE)
        logging.info("Using CLAM-SB classifier.")
    elif model_type == "camil":
        model = CAMILClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS,
            n_classes=2,
            agg_dim=int(getattr(config, "CAMIL_AGG_DIM", 512)),
            camil_n_layers=int(getattr(config, "CAMIL_N_LAYERS", 4)),
            temperature=float(getattr(config, "CAMIL_TEMPERATURE", 1.2)),
            dropout=float(getattr(config, "CAMIL_DROPOUT", 0.15)),
            gate=bool(getattr(config, "CAMIL_GATE", True)),
            attention_dim=int(getattr(config, "CAMIL_ATTENTION_DIM", 256)),
        ).to(config.DEVICE)
        logging.info("Using CAMIL classifier.")
    elif model_type == "rrtmil":
        model = RRTMILClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS,
            n_classes=2,
            rrt_mlp_dim=int(getattr(config, "RRT_MLP_DIM", 512)),
            rrt_n_layers=int(getattr(config, "RRT_N_LAYERS", 2)),
            rrt_n_heads=int(getattr(config, "RRT_N_HEADS", 8)),
            rrt_region_num=int(getattr(config, "RRT_REGION_NUM", 8)),
            rrt_dropout=float(getattr(config, "RRT_DROPOUT", 0.25)),
            rrt_act=str(getattr(config, "RRT_ACT", "relu")),
            rrt_attn=str(getattr(config, "RRT_ATTN", "rmsa")),
            rrt_pool=str(getattr(config, "RRT_POOL", "attn")),
            rrt_da_act=str(getattr(config, "RRT_DA_ACT", "relu")),
            rrt_trans_dropout=float(getattr(config, "RRT_TRANS_DROPOUT", 0.1)),
            rrt_drop_path=float(getattr(config, "RRT_DROP_PATH", 0.0)),
            rrt_epeg=bool(getattr(config, "RRT_EPEG", True)),
            rrt_epeg_k=int(getattr(config, "RRT_EPEG_K", 15)),
            rrt_cr_msa=bool(getattr(config, "RRT_CR_MSA", True)),
            rrt_crmsa_k=int(getattr(config, "RRT_CRMSA_K", 3)),
            rrt_crmsa_heads=int(getattr(config, "RRT_CRMSA_HEADS", 8)),
            rrt_all_shortcut=bool(getattr(config, "RRT_ALL_SHORTCUT", False)),
            rrt_crmsa_mlp=bool(getattr(config, "RRT_CRMSA_MLP", False)),
            rrt_qkv_bias=bool(getattr(config, "RRT_QKV_BIAS", True)),
            rrt_min_region_num=int(getattr(config, "RRT_MIN_REGION_NUM", 0)),
            rrt_trans_dim=int(getattr(config, "RRT_TRANS_DIM", 64)),
            rrt_ffn=bool(getattr(config, "RRT_FFN", False)),
            rrt_mlp_ratio=float(getattr(config, "RRT_MLP_RATIO", 4.0)),
        ).to(config.DEVICE)
        logging.info("Using RRT-MIL classifier.")
    else:
        model = MILClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS
        ).to(config.DEVICE)
        logging.info("Using current asymmetric MIL classifier.")

    # --- 3. 训练设置 ---
    # 定义两个criterion，因为它们的reduction方式不同
    criterion_sum = nn.CrossEntropyLoss(reduction='sum') # 或者使用 .sum() 手动计算
    criterion_mean = nn.CrossEntropyLoss(reduction='mean')
    # optimizer = optim.AdamW(model.parameters(), lr=config.CLS_LR,)
    optimizer = optim.AdamW(model.parameters(), lr=lr,)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)

    # --- 4. 训练循环与“双轨非对称”损失计算 ---
    best_val_auc = -1.0
    best_model_info = {}
    fold_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}")
    best_model_path = os.path.join(fold_dir, "best_mil+_classifier_model.pth")
    
    logging.info(f"Starting MIL training for {config.CLS_EPOCHS} epochs...")
    for epoch in range(config.CLS_EPOCHS):
        model.train()
        
        # 初始化用于记录的累加器
        epoch_total_loss = 0.0
        epoch_l1_plus, epoch_l2_plus, epoch_l1_minus, epoch_l2_minus = 0.0, 0.0, 0.0, 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.CLS_EPOCHS} [Training]")
        for batch in progress_bar:
            batch = {k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # --- a. 检查与解包 ---
            if 'sequences' not in batch or batch['sequences'].shape[0] == 0: continue
            
            optimizer.zero_grad()
            
            # --- a. 获取模型输出 ---
            predictions = model(batch)
            bag_logits = predictions.get('bag_logits')
            instance_logits = predictions.get('instance_logits')
            patient_logits = predictions.get('patient_logits')
            patient_bag_logits = predictions.get('patient_bag_logits')
            patient_max_logits = predictions.get('patient_max_logits')
            tier1_logits = predictions.get('tier1_logits')
            tier1_patient_indices = predictions.get('tier1_patient_indices')
            clam_instance_loss = predictions.get('clam_instance_loss')
            
            if (bag_logits is None or bag_logits.shape[0] == 0) and (
                patient_logits is None or patient_logits.shape[0] == 0
            ):
                continue

            # --- b. 调用你编写的两个损失函数 ---
            w_plus=w_plus
            dsmil_bag_w = float(getattr(config, "DSMIL_LOSS_BAG_WEIGHT", 0.5))
            dsmil_max_w = float(getattr(config, "DSMIL_LOSS_MAX_WEIGHT", 0.5))
            dtfd_t1_w = float(getattr(config, "DTFD_LOSS_TIER1_WEIGHT", 1.0))
            dtfd_t2_w = float(getattr(config, "DTFD_LOSS_TIER2_WEIGHT", 1.0))
            clam_bag_w = float(getattr(config, "CLAM_BAG_WEIGHT", 0.7))
            clam_no_inst = bool(getattr(config, "CLAM_NO_INST_CLUSTER", False))
            # DSMIL 使用官方训练目标: 0.5*CE(max)+0.5*CE(bag)
            if (
                model_type == "dsmil"
                and patient_bag_logits is not None
                and patient_max_logits is not None
                and patient_bag_logits.shape[0] == batch['labels'].shape[0]
                and patient_max_logits.shape[0] == batch['labels'].shape[0]
            ):
                loss_bag = criterion_mean(patient_bag_logits, batch['labels'])
                loss_max = criterion_mean(patient_max_logits, batch['labels'])
                loss_patient_level = dsmil_bag_w * loss_bag + dsmil_max_w * loss_max
                with torch.no_grad():
                    pos_mask = batch['labels'] == 1
                    neg_mask = batch['labels'] == 0
                    loss_pos = criterion_mean(patient_bag_logits[pos_mask], batch['labels'][pos_mask]) if torch.any(pos_mask) else torch.tensor(0.0, device=config.DEVICE)
                    loss_neg = criterion_mean(patient_bag_logits[neg_mask], batch['labels'][neg_mask]) if torch.any(neg_mask) else torch.tensor(0.0, device=config.DEVICE)
            elif (
                model_type == "dtfd"
                and patient_logits is not None
                and patient_logits.shape[0] == batch['labels'].shape[0]
            ):
                loss_t2 = criterion_mean(patient_logits, batch['labels'])
                if (
                    tier1_logits is not None
                    and tier1_patient_indices is not None
                    and tier1_logits.shape[0] > 0
                ):
                    t1_targets = batch['labels'][tier1_patient_indices]
                    loss_t1 = criterion_mean(tier1_logits, t1_targets)
                else:
                    loss_t1 = torch.tensor(0.0, device=config.DEVICE)
                loss_patient_level = dtfd_t1_w * loss_t1 + dtfd_t2_w * loss_t2
                with torch.no_grad():
                    pos_mask = batch['labels'] == 1
                    neg_mask = batch['labels'] == 0
                    loss_pos = criterion_mean(patient_logits[pos_mask], batch['labels'][pos_mask]) if torch.any(pos_mask) else torch.tensor(0.0, device=config.DEVICE)
                    loss_neg = criterion_mean(patient_logits[neg_mask], batch['labels'][neg_mask]) if torch.any(neg_mask) else torch.tensor(0.0, device=config.DEVICE)
            elif (
                model_type == "clam"
                and patient_logits is not None
                and patient_logits.shape[0] == batch['labels'].shape[0]
            ):
                bag_loss = criterion_mean(patient_logits, batch['labels'])
                if clam_no_inst:
                    loss_patient_level = bag_loss
                else:
                    inst_loss = clam_instance_loss if clam_instance_loss is not None else torch.tensor(0.0, device=config.DEVICE)
                    loss_patient_level = clam_bag_w * bag_loss + (1.0 - clam_bag_w) * inst_loss
                with torch.no_grad():
                    pos_mask = batch['labels'] == 1
                    neg_mask = batch['labels'] == 0
                    loss_pos = criterion_mean(patient_logits[pos_mask], batch['labels'][pos_mask]) if torch.any(pos_mask) else torch.tensor(0.0, device=config.DEVICE)
                    loss_neg = criterion_mean(patient_logits[neg_mask], batch['labels'][neg_mask]) if torch.any(neg_mask) else torch.tensor(0.0, device=config.DEVICE)
            # ABMIL 使用病人级logits直接监督；否则沿用现有不对称MIL损失
            elif patient_logits is not None and patient_logits.shape[0] == batch['labels'].shape[0]:
                loss_patient_level = criterion_mean(patient_logits, batch['labels'])
                with torch.no_grad():
                    pos_mask = batch['labels'] == 1
                    neg_mask = batch['labels'] == 0
                    loss_pos = criterion_mean(patient_logits[pos_mask], batch['labels'][pos_mask]) if torch.any(pos_mask) else torch.tensor(0.0, device=config.DEVICE)
                    loss_neg = criterion_mean(patient_logits[neg_mask], batch['labels'][neg_mask]) if torch.any(neg_mask) else torch.tensor(0.0, device=config.DEVICE)
            else:
                loss_patient_level, loss_pos, loss_neg = compute_mil_loss(
                    bag_logits,
                    batch['labels'],
                    batch['component_patient_indices'],
                    w_plus=w_plus,
                    w_minus=1.0
                )
            
            # 计算ROI损失 (你称之为 L2 Loss)
            if bag_logits is not None and instance_logits is not None and instance_logits.shape[0] > 0:
                batch_l2_plus, batch_l2_minus = compute_roi_loss(
                    bag_logits,
                    instance_logits,
                    batch['labels'],
                    batch['component_patient_indices'],
                    batch['instance_patient_indices'],
                    batch['instance_component_indices'],
                    criterion_mean
                )
            else:
                batch_l2_plus, batch_l2_minus = [], []
            
            # --- c. 聚合ROI损失 ---
            loss_l2_plus = torch.sum(torch.stack(batch_l2_plus)) if batch_l2_plus else torch.tensor(0.0, device=config.DEVICE)
            loss_l2_minus = torch.sum(torch.stack(batch_l2_minus)) if batch_l2_minus else torch.tensor(0.0, device=config.DEVICE)
            
            loss_roi_level = loss_l2_plus + loss_l2_minus

            # --- d. 组合总损失 ---
            # 根据你的描述，总损失是这两个的组合
            # 你需要一个权重来平衡它们
            # **【关键】** 设置为0时，就只使用你的病人损失
            lambda_roi_loss = roi
            total_loss = loss_patient_level + lambda_roi_loss * loss_roi_level
            
            # --- 损失计算结束 ---

            if total_loss > 0:
                total_loss.backward()
                optimizer.step()
            
            # 累加用于日志记录
            epoch_total_loss += total_loss.item()
            epoch_l1_plus += loss_pos.item()
            epoch_l2_plus += loss_l2_plus.item()
            epoch_l1_minus += loss_neg.item()
            epoch_l2_minus += loss_l2_minus.item()

            progress_bar.set_postfix(loss=total_loss.item())
            logging.info(f'epoch_l1_plus{epoch_l1_plus}')
            logging.info(f'epoch_l1_minus{epoch_l1_minus}')
        # --- 在 epoch 结束后，计算平均值并记录 ---
        num_batches = len(train_loader)
        training_history['train_loss'].append(epoch_total_loss / num_batches)
        training_history['loss_positive'].append((epoch_l1_plus + epoch_l2_plus) / num_batches) # 合并恶性损失
        training_history['loss_negative'].append((epoch_l1_minus + epoch_l2_minus) / num_batches) # 合并良性损失
        # logging.info("开始验证")
        val_metrics = evaluate_classifier(model, val_loader, device=config.DEVICE, is_mil=True)
        training_history['val_loss'].append(val_metrics.get('loss', 0.0)) # 使用.get()避免KeyError
        training_history['val_auc'].append(val_metrics['auc'])
        
        scheduler.step(val_metrics['auc'])
        
        logging.info(f"Epoch {epoch+1} | Train Loss: {training_history['train_loss'][-1]:.4f} | Val Loss: {training_history['val_loss'][-1]:.4f} | Val AUC: {training_history['val_auc'][-1]:.4f}")
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f" LR: {current_lr:.6f}")
        logging.info(f'lambda_roi_loss{lambda_roi_loss}')
        logging.info(f'w_plus{w_plus}')
        
        # 将 if 判断移到循环内部
        if val_metrics['auc'] > best_val_auc:
            best_metrics = val_metrics
            best_val_auc = val_metrics['auc']
            deep_copied_state_dict = {k: v.clone().detach() for k, v in model.state_dict().items()}
            logging.info(f"🎉 New best validation AUC: {best_val_auc:.4f}. Saving model...")
            best_model_info = { 'model_state_dict': deep_copied_state_dict, 'optimal_threshold': val_metrics['optimal_threshold'], 'val_auc': best_val_auc }
            torch.save(best_model_info, best_model_path)

    
    logging.info("val metrics")
    for key, value in best_metrics.items():
        logging.info(f"{key.capitalize():<15}: {value:.4f}")
    # ... (训练结束后的绘图和最终日志打印) ...
    return best_model_path



# In modules/classification_pipeline.py

# def evaluate_classifier(model, loader, device, is_mil=False):
#     """
#     【最终版 - 基于CLS Token】
#     评估函数。推理逻辑基于病人内部“最大连通分量（CLS）概率”。
#     同时计算一个与此逻辑一致的验证损失。
#     """
#     model.eval()
#     total_val_loss = 0.0
#     all_targets, all_pred_probs = [], []
#     bag_criterion = nn.CrossEntropyLoss(reduction='sum') # 用于计算损失

#     with torch.no_grad():
#         for batch in loader:
#             batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
#             targets = batch['labels']
#             component_patient_indices = batch.get('component_patient_indices')

#             # 初始化，用于收集每个病人的最终预测概率
#             batch_final_probs = []
#             # 初始化，用于收集每个病人的损失
#             per_patient_losses = []

#             if 'sequences' in batch and batch['sequences'].shape[0] > 0 and is_mil:
#                 # --- 1. 获取模型输出 ---
#                 predictions = model(batch)
#                 bag_logits = predictions['bag_logits'] # (Total_Components, 2)
                
#                 if bag_logits is not None and bag_logits.shape[0] > 0:
#                     bag_probs_pos = F.softmax(bag_logits, dim=1)[:, 1] # (Total_Components,)

#                     # --- 2. 遍历每个病人，计算其最终概率和损失 ---
#                     for i in range(batch['num_patients']):
#                         patient_label = targets[i]
#                         is_patient_i_component_mask = (component_patient_indices == i)
                        
#                         if not torch.any(is_patient_i_component_mask):
#                             # 病人无连通分量，赋予中性概率，损失为0
#                             batch_final_probs.append(torch.tensor(0.5, device=device))
#                             continue
                        
#                         patient_component_logits = bag_logits[is_patient_i_component_mask]
#                         patient_component_probs = bag_probs_pos[is_patient_i_component_mask]

#                         # a. 计算最终预测概率 (推理逻辑)
#                         max_prob, _ = torch.max(patient_component_probs, dim=0)
#                         batch_final_probs.append(max_prob)

#                         # b. 计算验证损失 (与训练时的 l1+/l1- 逻辑一致)
#                         if patient_label == 1: # 恶性
#                             _, champion_local_idx = torch.max(patient_component_logits[:, 1], dim=0)
#                             champion_logit = patient_component_logits[champion_local_idx].unsqueeze(0)
#                             loss = bag_criterion(champion_logit, torch.ones(1, dtype=torch.long, device=device))
#                         else: # 良性
#                             negative_labels = torch.zeros(patient_component_logits.shape[0], dtype=torch.long, device=device)
#                             loss = bag_criterion(patient_component_logits, negative_labels)
                        
#                         per_patient_losses.append(loss)
                
#                 probs_tensor = torch.stack(batch_final_probs)

#             elif batch['num_patients'] > 0: # 批次中有病人但无序列
#                 probs_tensor = torch.full((batch['num_patients'],), 0.5, device=device)
#             else: # 空批次
#                 continue
            
#             # --- 3. 累加损失和结果 ---
#             if per_patient_losses:
#                 batch_loss = torch.mean(torch.stack(per_patient_losses))
#                 total_val_loss += batch_loss.item()
            
#             all_targets.extend(targets.cpu().numpy())
#             all_pred_probs.extend(probs_tensor.cpu().numpy())

#     # --- 后续指标计算完全不变 ---
#     avg_loss = total_val_loss / len(loader) if len(loader) > 0 else 0.0
#     metrics = {'loss': avg_loss}
    
#     if len(np.unique(all_targets)) > 1:
#         metrics['auc'] = roc_auc_score(all_targets, all_pred_probs)
#         fpr, tpr, thresholds = roc_curve(all_targets, all_pred_probs)
#         optimal_idx = np.argmax(tpr - fpr)
#         optimal_threshold = thresholds[optimal_idx]
#     else:
#         metrics['auc'] = 0.5
#         optimal_threshold = 0.5
        
#     # ... (accuracy, sensitivity 等计算，使用 self-determined optimal_threshold)
    
#     metrics['optimal_threshold'] = optimal_threshold
#     binary_preds = (np.array(all_pred_probs) >= optimal_threshold).astype(int)
#     metrics['accuracy'] = accuracy_score(all_targets, binary_preds)
#     metrics['sensitivity'] = recall_score(all_targets, binary_preds, zero_division=0)
#     metrics['precision'] = precision_score(all_targets, binary_preds, zero_division=0)
#     metrics['f1_score'] = f1_score(all_targets, binary_preds, zero_division=0)
    
#     try:
#         tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds).ravel()
#         metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
#     except ValueError:
#         metrics['specificity'] = 0

#     return metrics
def evaluate_classifier(model, loader, device, is_mil=False):
    """
    【最终版 - 基于CLS Token】
    评估函数。推理逻辑基于病人内部"最大连通分量（CLS）概率"。
    同时计算一个与此逻辑一致的验证损失。
    """
    model.eval()
    model_type = getattr(config, "CLS_MODEL_TYPE", "mil_asym").lower()
    dsmil_bag_w = float(getattr(config, "DSMIL_LOSS_BAG_WEIGHT", 0.5))
    dsmil_max_w = float(getattr(config, "DSMIL_LOSS_MAX_WEIGHT", 0.5))
    dsmil_infer_use_bag = bool(getattr(config, "DSMIL_INFER_USE_BAG_LOGITS", True))
    dtfd_t1_w = float(getattr(config, "DTFD_LOSS_TIER1_WEIGHT", 1.0))
    dtfd_t2_w = float(getattr(config, "DTFD_LOSS_TIER2_WEIGHT", 1.0))
    clam_bag_w = float(getattr(config, "CLAM_BAG_WEIGHT", 0.7))
    clam_no_inst = bool(getattr(config, "CLAM_NO_INST_CLUSTER", False))
    total_val_loss = 0.0
    all_targets, all_pred_probs = [], []
    bag_criterion = nn.CrossEntropyLoss(reduction='sum') # 用于计算损失

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            targets = batch['labels']
            component_patient_indices = batch.get('component_patient_indices')

            # 初始化，用于收集每个病人的最终预测概率
            batch_final_probs = []
            # 初始化，用于收集每个病人的损失
            per_patient_losses = []

            if 'sequences' in batch and batch['sequences'].shape[0] > 0 and is_mil:
                # --- 1. 获取模型输出 ---
                predictions = model(batch)
                patient_logits = predictions.get('patient_logits')
                patient_bag_logits = predictions.get('patient_bag_logits')
                patient_max_logits = predictions.get('patient_max_logits')
                tier1_logits = predictions.get('tier1_logits')
                tier1_patient_indices = predictions.get('tier1_patient_indices')
                clam_instance_loss = predictions.get('clam_instance_loss')
                bag_logits = predictions.get('bag_logits') # (Total_Components, 2)

                if (
                    model_type == "clam"
                    and patient_logits is not None
                    and patient_logits.shape[0] == batch['num_patients']
                ):
                    probs_tensor = F.softmax(patient_logits, dim=1)[:, 1]
                    bag_loss = F.cross_entropy(patient_logits, targets, reduction='mean')
                    if clam_no_inst:
                        total_val_loss += bag_loss.item()
                    else:
                        inst_loss = clam_instance_loss if clam_instance_loss is not None else torch.tensor(0.0, device=device)
                        total_val_loss += (clam_bag_w * bag_loss + (1.0 - clam_bag_w) * inst_loss).item()
                    all_targets.extend(targets.cpu().numpy())
                    all_pred_probs.extend(probs_tensor.cpu().numpy())
                    continue

                if (
                    model_type == "dtfd"
                    and patient_logits is not None
                    and patient_logits.shape[0] == batch['num_patients']
                ):
                    probs_tensor = F.softmax(patient_logits, dim=1)[:, 1]
                    loss_t2 = F.cross_entropy(patient_logits, targets, reduction='mean')
                    if (
                        tier1_logits is not None
                        and tier1_patient_indices is not None
                        and tier1_logits.shape[0] > 0
                    ):
                        t1_targets = targets[tier1_patient_indices]
                        loss_t1 = F.cross_entropy(tier1_logits, t1_targets, reduction='mean')
                    else:
                        loss_t1 = torch.tensor(0.0, device=device)
                    total_val_loss += (dtfd_t1_w * loss_t1 + dtfd_t2_w * loss_t2).item()
                    all_targets.extend(targets.cpu().numpy())
                    all_pred_probs.extend(probs_tensor.cpu().numpy())
                    continue

                # DSMIL 路径：按配置选择推理分支，并用配置权重计算验证损失
                if (
                    model_type == "dsmil"
                    and
                    patient_bag_logits is not None
                    and patient_max_logits is not None
                    and patient_bag_logits.shape[0] == batch['num_patients']
                    and patient_max_logits.shape[0] == batch['num_patients']
                ):
                    if dsmil_infer_use_bag:
                        infer_logits = patient_bag_logits
                    else:
                        infer_logits = predictions.get('patient_logits')
                        if infer_logits is None or infer_logits.shape[0] != batch['num_patients']:
                            infer_logits = dsmil_bag_w * patient_bag_logits + dsmil_max_w * patient_max_logits

                    probs_tensor = F.softmax(infer_logits, dim=1)[:, 1]
                    loss_bag = F.cross_entropy(patient_bag_logits, targets, reduction='mean')
                    loss_max = F.cross_entropy(patient_max_logits, targets, reduction='mean')
                    total_val_loss += (dsmil_bag_w * loss_bag + dsmil_max_w * loss_max).item()
                    all_targets.extend(targets.cpu().numpy())
                    all_pred_probs.extend(probs_tensor.cpu().numpy())
                    continue

                # ABMIL路径：直接使用病人级logits
                if patient_logits is not None and patient_logits.shape[0] == batch['num_patients']:
                    probs_tensor = F.softmax(patient_logits, dim=1)[:, 1]
                    batch_loss = F.cross_entropy(patient_logits, targets, reduction='mean')
                    total_val_loss += batch_loss.item()
                    all_targets.extend(targets.cpu().numpy())
                    all_pred_probs.extend(probs_tensor.cpu().numpy())
                    continue
                
                if bag_logits is not None and bag_logits.shape[0] > 0:
                    bag_probs_pos = F.softmax(bag_logits, dim=1)[:, 1] # (Total_Components,)

                    # --- 2. 遍历每个病人，计算其最终概率和损失 ---
                    for i in range(batch['num_patients']):
                        patient_label = targets[i]
                        is_patient_i_component_mask = (component_patient_indices == i)
                        
                        if not torch.any(is_patient_i_component_mask):
                            # 病人无连通分量，赋予中性概率，损失为0
                            batch_final_probs.append(torch.tensor(0.5, device=device))
                            continue
                        
                        patient_component_logits = bag_logits[is_patient_i_component_mask]
                        patient_component_probs = bag_probs_pos[is_patient_i_component_mask]

                        # a. 计算最终预测概率 (推理逻辑)
                        max_prob, _ = torch.max(patient_component_probs, dim=0)
                        batch_final_probs.append(max_prob)

                        # b. 计算验证损失 (与训练时的 l1+/l1- 逻辑一致)
                        if patient_label == 1: # 恶性
                            _, champion_local_idx = torch.max(patient_component_logits[:, 1], dim=0)
                            champion_logit = patient_component_logits[champion_local_idx].unsqueeze(0)
                            loss = bag_criterion(champion_logit, torch.ones(1, dtype=torch.long, device=device))
                        else: # 良性
                            negative_labels = torch.zeros(patient_component_logits.shape[0], dtype=torch.long, device=device)
                            loss = bag_criterion(patient_component_logits, negative_labels)
                        
                        per_patient_losses.append(loss)
                
                probs_tensor = torch.stack(batch_final_probs)

            elif batch['num_patients'] > 0: # 批次中有病人但无序列
                probs_tensor = torch.full((batch['num_patients'],), 0.5, device=device)
            else: # 空批次
                continue
            
            # --- 3. 累加损失和结果 ---
            if per_patient_losses:
                batch_loss = torch.mean(torch.stack(per_patient_losses))
                total_val_loss += batch_loss.item()
            
            all_targets.extend(targets.cpu().numpy())
            all_pred_probs.extend(probs_tensor.cpu().numpy())

    # --- 后续指标计算，修改阈值选择策略为基于F1-score最大 ---
    avg_loss = total_val_loss / len(loader) if len(loader) > 0 else 0.0
    metrics = {'loss': avg_loss}
    
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
    metrics['max_f1'] = max_f1
    
    # 使用最优阈值计算最终预测
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

# def evaluate_classifier(model, loader, device, is_mil=False):
#     """
#     【最终版 - 基于CLS Token】
#     评估函数。使用病人预测概率作为候选阈值来选择最优阈值。
#     """
#     model.eval()
#     total_val_loss = 0.0
#     all_targets, all_pred_probs = [], []
#     bag_criterion = nn.CrossEntropyLoss(reduction='sum')

#     with torch.no_grad():
#         for batch in loader:
#             batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
#             targets = batch['labels']
#             component_patient_indices = batch.get('component_patient_indices')

#             batch_final_probs = []
#             per_patient_losses = []

#             if 'sequences' in batch and batch['sequences'].shape[0] > 0 and is_mil:
#                 predictions = model(batch)
#                 bag_logits = predictions['bag_logits']
                
#                 if bag_logits is not None and bag_logits.shape[0] > 0:
#                     bag_probs_pos = F.softmax(bag_logits, dim=1)[:, 1]

#                     for i in range(batch['num_patients']):
#                         patient_label = targets[i]
#                         is_patient_i_component_mask = (component_patient_indices == i)
                        
#                         if not torch.any(is_patient_i_component_mask):
#                             batch_final_probs.append(torch.tensor(0.5, device=device))
#                             continue
                        
#                         patient_component_logits = bag_logits[is_patient_i_component_mask]
#                         patient_component_probs = bag_probs_pos[is_patient_i_component_mask]

#                         # 计算最终预测概率
#                         max_prob, _ = torch.max(patient_component_probs, dim=0)
#                         batch_final_probs.append(max_prob)

#                         # 计算验证损失
#                         if patient_label == 1:
#                             _, champion_local_idx = torch.max(patient_component_logits[:, 1], dim=0)
#                             champion_logit = patient_component_logits[champion_local_idx].unsqueeze(0)
#                             loss = bag_criterion(champion_logit, torch.ones(1, dtype=torch.long, device=device))
#                         else:
#                             negative_labels = torch.zeros(patient_component_logits.shape[0], dtype=torch.long, device=device)
#                             loss = bag_criterion(patient_component_logits, negative_labels)
                        
#                         per_patient_losses.append(loss)
                
#                 probs_tensor = torch.stack(batch_final_probs)

#             elif batch['num_patients'] > 0:
#                 probs_tensor = torch.full((batch['num_patients'],), 0.5, device=device)
#             else:
#                 continue
            
#             if per_patient_losses:
#                 batch_loss = torch.mean(torch.stack(per_patient_losses))
#                 total_val_loss += batch_loss.item()
            
#             all_targets.extend(targets.cpu().numpy())
#             all_pred_probs.extend(probs_tensor.cpu().numpy())

#     # --- 使用病人预测概率作为候选阈值 ---
#     avg_loss = total_val_loss / len(loader) if len(loader) > 0 else 0.0
#     metrics = {'loss': avg_loss}
    
#     if len(np.unique(all_targets)) > 1:
#         metrics['auc'] = roc_auc_score(all_targets, all_pred_probs)
        
#         # 使用所有病人的预测概率作为候选阈值，并添加边界值
#         candidate_thresholds = np.unique(all_pred_probs)
#         # 添加0和1作为边界阈值，确保覆盖完整范围
#         candidate_thresholds = np.sort(np.concatenate([candidate_thresholds, [0.0, 1.0]]))
        
#         # 去除重复并排序
#         candidate_thresholds = np.unique(candidate_thresholds)
        
#         f1_scores = []
#         for threshold in candidate_thresholds:
#             binary_preds = (np.array(all_pred_probs) >= threshold).astype(int)
#             f1 = f1_score(all_targets, binary_preds, zero_division=0)
#             f1_scores.append(f1)
        
#         # 找到F1-score最大的阈值
#         optimal_idx = np.argmax(f1_scores)
#         optimal_threshold = candidate_thresholds[optimal_idx]
#         max_f1 = f1_scores[optimal_idx]
        
#         # 打印调试信息
#         logging.info(f"候选阈值数量: {len(candidate_thresholds)}")
#         logging.info(f"最优阈值: {optimal_threshold:.4f}, 最大F1: {max_f1:.4f}")
        
#     else:
#         metrics['auc'] = 0.5
#         optimal_threshold = 0.5
#         max_f1 = 0.0
        
#     metrics['optimal_threshold'] = optimal_threshold
#     metrics['max_f1'] = max_f1
    
#     # 使用最优阈值计算最终预测
#     binary_preds = (np.array(all_pred_probs) >= optimal_threshold).astype(int)
#     metrics['accuracy'] = accuracy_score(all_targets, binary_preds)
#     metrics['sensitivity'] = recall_score(all_targets, binary_preds, zero_division=0)
#     metrics['precision'] = precision_score(all_targets, binary_preds, zero_division=0)
#     metrics['f1_score'] = f1_score(all_targets, binary_preds, zero_division=0)
    
#     try:
#         tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds).ravel()
#         metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
#     except ValueError:
#         metrics['specificity'] = 0

#     return metrics

from models_mil import MILClassifier # <-- 确保你导入了新的MIL模型

def run_classification_testing(
    fold_num: int, 
    test_pids: List[str],
    segmentation_model_path: str,
    classifier_model_path: str
) -> Dict[str, Any]:
    """
    【最终版】在测试集上评估最终的、基于MIL-Attention的分类模型。
    """
    logging.info(f"========== Starting MIL-Attention Classification Testing for Fold {fold_num} ==========")
    
    # 1. 加载用于特征提取的模型 (逻辑不变)
    logging.info("Loading models for feature extraction...")
    from dinov2_loader import load_dinov2_base_encoder

    base_encoder = load_dinov2_base_encoder(
        config.DINOV2_REPO_PATH, config.DINOV2_PRETRAINED_WEIGHTS
    )
    segmentation_model = DINOV2EncoderLoRA(
        encoder=base_encoder, r=config.SEG_R_LORA, emb_dim=config.SEG_EMB_DIM, img_dim=config.SEG_IMG_DIM,
        n_classes=config.SEG_N_CLASSES, use_lora=True, use_fpn=False
    ).to(config.DEVICE)
    segmentation_model.load_parameters(segmentation_model_path)
    feature_backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-4]).to(config.DEVICE)

    # 2. 提取测试集特征并创建DataLoader (逻辑不变)
    test_data = extract_features_for_pids(
        test_pids, 
        segmentation_model, 
        feature_backbone, 
        fold_num, 
        'test', 
        class_aug_ratio={0: 1, 1: 1}
    )
    test_ds = PatientComponentDataset(test_data)
    test_loader = DataLoader(
        test_ds, 
        batch_size=config.CLS_BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=patient_collate_fn
    )

    # 3. **【改动】** 初始化新的 MIL 分类模型并加载最佳权重
    logging.info("Initializing MIL Attention classifier model for testing...")
    model_type = getattr(config, "CLS_MODEL_TYPE", "mil_asym").lower()
    if model_type == "abmil":
        model = ABMILClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS,
            attention_dim=getattr(config, "ABMIL_ATTENTION_DIM", 128),
            gated=bool(getattr(config, "ABMIL_GATED", True)),
        ).to(config.DEVICE)
    elif model_type == "dsmil":
        model = DSMILClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS,
            q_dim=int(getattr(config, "DSMIL_Q_DIM", 128)),
            nonlinear=bool(getattr(config, "DSMIL_NONLINEAR", True)),
            passing_v=bool(getattr(config, "DSMIL_PASSING_V", False)),
            dropout_v=float(getattr(config, "DSMIL_DROPOUT_V", 0.0)),
        ).to(config.DEVICE)
    elif model_type == "transmil":
        model = TransMILClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS,
        ).to(config.DEVICE)
    elif model_type == "dtfd":
        model = DTFDClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS,
            n_classes=2,
            num_group=int(getattr(config, "DTFD_NUM_GROUP", 4)),
            total_instance=int(getattr(config, "DTFD_TOTAL_INSTANCE", 4)),
            distill_type=str(getattr(config, "DTFD_DISTILL_TYPE", "AFS")),
            num_res_layers=int(getattr(config, "DTFD_NUM_RES_LAYERS", 0)),
            droprate_tier1=float(getattr(config, "DTFD_DROPRATE_TIER1", 0.0)),
            droprate_tier2=float(getattr(config, "DTFD_DROPRATE_TIER2", 0.0)),
            attn_dim=int(getattr(config, "DTFD_ATTN_DIM", 128)),
        ).to(config.DEVICE)
    elif model_type == "clam":
        model = CLAMSBClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS,
            n_classes=2,
            gate=bool(getattr(config, "CLAM_GATE", True)),
            attn_hidden_dim=int(getattr(config, "CLAM_ATTN_HIDDEN_DIM", 256)),
            dropout=float(getattr(config, "CLAM_DROPOUT", 0.25)),
            k_sample=int(getattr(config, "CLAM_K_SAMPLE", 8)),
            subtyping=bool(getattr(config, "CLAM_SUBTYPING", False)),
            no_inst_cluster=bool(getattr(config, "CLAM_NO_INST_CLUSTER", False)),
            inst_loss_type=str(getattr(config, "CLAM_INST_LOSS", "ce")),
        ).to(config.DEVICE)
    elif model_type == "camil":
        model = CAMILClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS,
            n_classes=2,
            agg_dim=int(getattr(config, "CAMIL_AGG_DIM", 512)),
            camil_n_layers=int(getattr(config, "CAMIL_N_LAYERS", 4)),
            temperature=float(getattr(config, "CAMIL_TEMPERATURE", 1.2)),
            dropout=float(getattr(config, "CAMIL_DROPOUT", 0.15)),
            gate=bool(getattr(config, "CAMIL_GATE", True)),
            attention_dim=int(getattr(config, "CAMIL_ATTENTION_DIM", 256)),
        ).to(config.DEVICE)
    elif model_type == "rrtmil":
        model = RRTMILClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM,
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS,
            n_layers=config.CLS_N_LAYERS,
            n_classes=2,
            rrt_mlp_dim=int(getattr(config, "RRT_MLP_DIM", 512)),
            rrt_n_layers=int(getattr(config, "RRT_N_LAYERS", 2)),
            rrt_n_heads=int(getattr(config, "RRT_N_HEADS", 8)),
            rrt_region_num=int(getattr(config, "RRT_REGION_NUM", 8)),
            rrt_dropout=float(getattr(config, "RRT_DROPOUT", 0.25)),
            rrt_act=str(getattr(config, "RRT_ACT", "relu")),
            rrt_attn=str(getattr(config, "RRT_ATTN", "rmsa")),
            rrt_pool=str(getattr(config, "RRT_POOL", "attn")),
            rrt_da_act=str(getattr(config, "RRT_DA_ACT", "relu")),
            rrt_trans_dropout=float(getattr(config, "RRT_TRANS_DROPOUT", 0.1)),
            rrt_drop_path=float(getattr(config, "RRT_DROP_PATH", 0.0)),
            rrt_epeg=bool(getattr(config, "RRT_EPEG", True)),
            rrt_epeg_k=int(getattr(config, "RRT_EPEG_K", 15)),
            rrt_cr_msa=bool(getattr(config, "RRT_CR_MSA", True)),
            rrt_crmsa_k=int(getattr(config, "RRT_CRMSA_K", 3)),
            rrt_crmsa_heads=int(getattr(config, "RRT_CRMSA_HEADS", 8)),
            rrt_all_shortcut=bool(getattr(config, "RRT_ALL_SHORTCUT", False)),
            rrt_crmsa_mlp=bool(getattr(config, "RRT_CRMSA_MLP", False)),
            rrt_qkv_bias=bool(getattr(config, "RRT_QKV_BIAS", True)),
            rrt_min_region_num=int(getattr(config, "RRT_MIN_REGION_NUM", 0)),
            rrt_trans_dim=int(getattr(config, "RRT_TRANS_DIM", 64)),
            rrt_ffn=bool(getattr(config, "RRT_FFN", False)),
            rrt_mlp_ratio=float(getattr(config, "RRT_MLP_RATIO", 4.0)),
        ).to(config.DEVICE)
    else:
        model = MILClassifier(
            input_dim=config.BACKBONE_OUTPUT_DIM, 
            hidden_dim=config.CLS_HIDDEN_DIM,
            n_heads=config.CLS_N_HEADS, 
            n_layers=config.CLS_N_LAYERS,
            attention_dim=128
        ).to(config.DEVICE)
    
    checkpoint = torch.load(classifier_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimal_threshold_from_val = checkpoint['optimal_threshold']
    dsmil_infer_use_bag = bool(getattr(config, "DSMIL_INFER_USE_BAG_LOGITS", True))
    
    logging.info(f"Loaded best MIL classifier model from training. Using optimal threshold from validation set: {optimal_threshold_from_val:.4f}")

    # 4. **【改动】** 在测试集上评估
    model.eval()
    all_targets, all_pred_probs, all_pids = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing on Fold {fold_num}"):
            pids_in_batch = batch['pids']
            targets = batch['labels']
            component_patient_indices = batch.get('component_patient_indices')
            
            batch = {k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # --- 推理逻辑：计算每个病人的最大 CLS token 概率 ---
            batch_final_probs = []
            if 'sequences' in batch and batch['sequences'].shape[0] > 0:
                predictions = model(batch)
                patient_logits = predictions.get('patient_logits')
                patient_bag_logits = predictions.get('patient_bag_logits')
                bag_logits = predictions.get('bag_logits') # (Total_Components, 2)

                # DSMIL路径：严格按配置决定最终推理分支
                if (
                    model_type == "dsmil"
                    and patient_bag_logits is not None
                    and patient_bag_logits.shape[0] == batch['num_patients']
                ):
                    if dsmil_infer_use_bag:
                        infer_logits = patient_bag_logits
                    else:
                        infer_logits = patient_logits if patient_logits is not None and patient_logits.shape[0] == batch['num_patients'] else patient_bag_logits
                    probs_tensor = F.softmax(infer_logits, dim=1)[:, 1]
                    all_targets.extend(targets.cpu().numpy())
                    all_pred_probs.extend(probs_tensor.cpu().numpy())
                    all_pids.extend(pids_in_batch)
                    continue

                # ABMIL路径：直接读取病人级预测
                if patient_logits is not None and patient_logits.shape[0] == batch['num_patients']:
                    probs_tensor = F.softmax(patient_logits, dim=1)[:, 1]
                    all_targets.extend(targets.cpu().numpy())
                    all_pred_probs.extend(probs_tensor.cpu().numpy())
                    all_pids.extend(pids_in_batch)
                    continue
                
                if bag_logits is not None and bag_logits.shape[0] > 0:
                    bag_probs_pos = F.softmax(bag_logits, dim=1)[:, 1] # (Total_Components,)

                    for i in range(batch['num_patients']):
                        is_patient_i_component_mask = (component_patient_indices == i)
                        if torch.any(is_patient_i_component_mask):
                            patient_component_probs = bag_probs_pos[is_patient_i_component_mask]
                            max_prob, _ = torch.max(patient_component_probs, dim=0)
                            batch_final_probs.append(max_prob)
                        else:
                            batch_final_probs.append(torch.tensor(0.5, device=config.DEVICE))
                    
                    probs_tensor = torch.stack(batch_final_probs)
                else:
                    probs_tensor = torch.full((batch['num_patients'],), 0.5, device=config.DEVICE)
            
            elif batch['num_patients'] > 0:
                probs_tensor = torch.full((batch['num_patients'],), 0.5, device=config.DEVICE)
            else:
                continue
            
            # --- 收集结果 ---
            all_targets.extend(targets.cpu().numpy())
            all_pred_probs.extend(probs_tensor.cpu().numpy())
            all_pids.extend(pids_in_batch)

    # --- 5. 使用固定的验证集阈值，计算并报告最终性能指标 ---
    all_pred_probs = np.array(all_pred_probs)
    all_targets = np.array(all_targets)
    binary_preds = (all_pred_probs >= optimal_threshold_from_val).astype(int)
    
    final_test_metrics = {}
    if len(np.unique(all_targets)) > 1:
        final_test_metrics['auc'] = roc_auc_score(all_targets, all_pred_probs)
    else:
        final_test_metrics['auc'] = 0.5
        
    final_test_metrics['accuracy'] = accuracy_score(all_targets, binary_preds)
    final_test_metrics['sensitivity'] = recall_score(all_targets, binary_preds, pos_label=1, zero_division=0)
    final_test_metrics['specificity'] = recall_score(all_targets, binary_preds, pos_label=0, zero_division=0)
    final_test_metrics['precision'] = precision_score(all_targets, binary_preds, zero_division=0)
    final_test_metrics['f1_score'] = f1_score(all_targets, binary_preds, zero_division=0)
    
    # --- 6. 打印详细报告 ---
    logging.info("--- Test Set Evaluation Report (at validation threshold) ---")
    for key, value in final_test_metrics.items():
        logging.info(f"{key.capitalize():<15}: {value:.4f}")
    
    logging.info("\n--- Misclassified Patients on Test Set ---")
    misclassified_found = False
    for i in range(len(all_pids)):
        if binary_preds[i] != all_targets[i]:
            logging.info(f"  - PID: {all_pids[i]}, True: {all_targets[i]}, Predicted: {binary_preds[i]}, Probability_for_1: {all_pred_probs[i]:.4f}")
            misclassified_found = True
    if not misclassified_found:
        logging.info("  ✅ No misclassifications on the test set.")

    logging.info(f"\nConfusion Matrix (at threshold {optimal_threshold_from_val:.4f}):\n{confusion_matrix(all_targets, binary_preds)}")
    logging.info(f"========== Final MIL Testing for Fold {fold_num} Finished ==========")
    
    return final_test_metrics

