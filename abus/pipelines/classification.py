# classification_pipeline.py - ABUS 2D 工作流 + 多 MIL 方法支持
import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from dino_finetune import DINOV2EncoderLoRA
from scipy.ndimage import label
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import roi_align
from tqdm import tqdm

import config
from methods.eval import compute_patient_level_loss, compute_roi_loss, evaluate_classifier
from methods.factory import build_classifier_model, get_model_type


class PatientComponentDataset(Dataset):
    def __init__(self, patient_data: List[Dict[str, Any]]):
        self.patient_data = patient_data

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, idx):
        return self.patient_data[idx]


def patient_collate_fn(batch: List[Dict[str, Any]]):
    all_sequences, all_masks, component_patient_indices = [], [], []
    instance_patient_indices, instance_component_indices = [], []
    labels, pids = [], []
    component_global_idx = 0

    for i, patient_sample in enumerate(batch):
        labels.append(patient_sample["label"])
        pids.append(patient_sample["pid"])
        if not patient_sample.get("sequences"):
            continue
        for seq in patient_sample["sequences"]:
            seq = seq.squeeze()
            if seq.ndim == 1:
                seq = seq.unsqueeze(0)
            if seq.shape[0] == 0:
                continue
            num_rois, c = seq.shape
            effective_len = min(num_rois, config.MAX_ROIS)
            if num_rois >= config.MAX_ROIS:
                padded_seq = seq[: config.MAX_ROIS]
                actual_num_rois = config.MAX_ROIS
            else:
                padding = torch.zeros(config.MAX_ROIS - num_rois, c, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding], dim=0)
                actual_num_rois = num_rois
            mask = torch.zeros(config.MAX_ROIS, dtype=torch.bool)
            mask[:effective_len] = True
            all_sequences.append(padded_seq)
            all_masks.append(mask)
            component_patient_indices.append(i)
            instance_patient_indices.extend([i] * actual_num_rois)
            instance_component_indices.extend([component_global_idx] * actual_num_rois)
            component_global_idx += 1

    batch_dict = {"labels": torch.tensor(labels, dtype=torch.long), "pids": pids, "num_patients": len(batch)}
    if all_sequences:
        batch_dict.update(
            {
                "sequences": torch.stack(all_sequences, dim=0),
                "masks": torch.stack(all_masks, dim=0),
                "component_patient_indices": torch.tensor(component_patient_indices, dtype=torch.long),
                "instance_patient_indices": torch.tensor(instance_patient_indices, dtype=torch.long),
                "instance_component_indices": torch.tensor(instance_component_indices, dtype=torch.long),
            }
        )
    else:
        batch_dict.update(
            {
                "sequences": torch.empty(0, config.MAX_ROIS, config.BACKBONE_OUTPUT_DIM),
                "masks": torch.empty(0, config.MAX_ROIS, dtype=torch.bool),
                "component_patient_indices": torch.empty(0, dtype=torch.long),
                "instance_patient_indices": torch.empty(0, dtype=torch.long),
                "instance_component_indices": torch.empty(0, dtype=torch.long),
            }
        )
    return batch_dict


def _load_segmentation_weights(segmentation_model: nn.Module, segmentation_model_path: str) -> None:
    """兼容两种分割权重格式：
    1) save_parameters() 保存: w_a_000, w_b_000, decoder.*
    2) state_dict() 保存: 完整 PyTorch 键名（segmentation_trainer 默认方式）
    """
    state = torch.load(segmentation_model_path, map_location=config.DEVICE)
    if isinstance(state, dict) and "w_a_000" in state:
        segmentation_model.load_parameters(segmentation_model_path)
        logging.info(f"Loaded segmentation weights via load_parameters: {segmentation_model_path}")
        return

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    missing, unexpected = segmentation_model.load_state_dict(state, strict=False)
    if missing:
        logging.warning(f"Segmentation load_state_dict missing keys (first 5): {missing[:5]}")
    if unexpected:
        logging.warning(f"Segmentation load_state_dict unexpected keys (first 5): {unexpected[:5]}")
    logging.info(f"Loaded segmentation weights via load_state_dict: {segmentation_model_path}")


def _load_feature_models(segmentation_model_path: str):
    from dinov2_loader import load_dinov2_base_encoder

    base_encoder = load_dinov2_base_encoder(
        config.DINOV2_REPO_PATH, config.DINOV2_PRETRAINED_WEIGHTS
    )
    segmentation_model = DINOV2EncoderLoRA(
        encoder=base_encoder,
        r=config.SEG_R_LORA,
        emb_dim=config.SEG_EMB_DIM,
        img_dim=(config.SEG_IMG_DIM, config.SEG_IMG_DIM),
        n_classes=config.SEG_N_CLASSES,
        use_lora=True,
        use_fpn=False,
    ).to(config.DEVICE)
    _load_segmentation_weights(segmentation_model, segmentation_model_path)
    feature_backbone = nn.Sequential(
        *list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-4]
    ).to(config.DEVICE)
    return segmentation_model, feature_backbone


def extract_features_from_slices(
    slice_info_list: List[Dict],
    segmentation_model: nn.Module,
    feature_backbone: nn.Module,
    fold_identifier: str,
    dataset_type: str,
    class_aug_ratio: Dict[int, int],
) -> List[Dict[str, Any]]:
    cache_dir = os.path.join(config.OUTPUT_DIR, str(fold_identifier), "features_2d_workflow")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_type}_features.pth")

    if os.path.exists(cache_path):
        logging.info(f"Loading cached {dataset_type} features from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    patient_slices = defaultdict(list)
    for slice_info in slice_info_list:
        patient_slices[slice_info["pid"]].append(slice_info)

    logging.info(f"Extracting features for {len(patient_slices)} cases ({dataset_type})...")
    device = config.DEVICE
    thr = config.DETECTION_THRESHOLD
    segmentation_model.eval()
    feature_backbone.eval()

    preprocess_geo_tf = A.Compose(
        [
            A.LongestMaxSize(max_size=config.SEG_IMG_DIM),
            A.PadIfNeeded(
                min_height=config.SEG_IMG_DIM,
                min_width=config.SEG_IMG_DIM,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
        ]
    )
    to_tensor_tf = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    aug_tf = A.ReplayCompose(
        [
            A.Rotate(limit=30, p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.RandomScale(scale_limit=0.2, p=0.2),
            A.LongestMaxSize(max_size=config.SEG_IMG_DIM),
            A.PadIfNeeded(
                min_height=config.SEG_IMG_DIM,
                min_width=config.SEG_IMG_DIM,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
            A.RandomGamma(gamma_limit=(70, 150), p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        is_check_shapes=False,
    )
    no_aug_tf = A.ReplayCompose(
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
        is_check_shapes=False,
    )

    final_data = []
    for pid, slices in tqdm(patient_slices.items(), desc=f"Processing {dataset_type}"):
        slices.sort(key=lambda x: x["image_path"])
        patient_label = slices[0]["label"]
        pred_masks = []
        images_for_feature_extraction = []

        for slice_info in slices:
            img_bgr = cv2.imread(slice_info["image_path"])
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_392_uint8 = preprocess_geo_tf(image=img_rgb)["image"]
            images_for_feature_extraction.append(img_392_uint8)
            tensor_input = to_tensor_tf(image=img_392_uint8)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = segmentation_model(tensor_input)
                prob = torch.softmax(logits, dim=1)[0, 1]
                pred_masks.append((prob > thr).cpu().numpy().astype(np.uint8))

        if not pred_masks:
            continue

        binary_3d = np.stack(pred_masks, axis=0)
        labeled_array, num_feats = label(binary_3d, structure=np.ones((3, 3, 3)))
        if num_feats == 0:
            continue

        k = class_aug_ratio.get(patient_label, 1) if dataset_type == "train" else 1
        current_aug_tf = aug_tf if dataset_type == "train" else no_aug_tf
        base_seed = hash(pid) & 0xFFFFFFFF

        for aug_idx in range(k):
            aug_seed = base_seed + aug_idx
            random.seed(aug_seed)
            np.random.seed(aug_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(aug_seed)

            component_feats = defaultdict(list)
            replay_state = None
            for i, img_np in enumerate(images_for_feature_extraction):
                id_map_slice = labeled_array[i].astype(np.int32)
                unique_ids = np.unique(id_map_slice)
                unique_ids = unique_ids[unique_ids != 0]
                if len(unique_ids) == 0:
                    continue
                if replay_state is None:
                    out = current_aug_tf(image=img_np, mask=id_map_slice)
                    replay_state = out["replay"]
                else:
                    out = A.ReplayCompose.replay(replay_state, image=img_np, mask=id_map_slice)

                aug_img_t = out["image"].unsqueeze(0).to(device)
                aug_id_map = out["mask"].cpu().numpy()
                with torch.no_grad():
                    feat_map = feature_backbone(aug_img_t)
                hf, wf = feat_map.shape[2], feat_map.shape[3]
                hi, wi = aug_img_t.shape[2], aug_img_t.shape[3]
                sy, sx = hf / hi, wf / wi

                for comp_id in np.unique(aug_id_map):
                    if comp_id == 0:
                        continue
                    comp_mask = (aug_id_map == comp_id).astype(np.uint8)
                    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    all_points = np.concatenate(contours, axis=0)
                    x, y, w, h = cv2.boundingRect(all_points)
                    box = torch.tensor([[x * sx, y * sy, (x + w) * sx, (y + h) * sy]], device=device)
                    with torch.no_grad():
                        roi_feat = roi_align(
                            feat_map,
                            torch.cat([torch.zeros(1, 1, device=device), box], dim=1),
                            output_size=(1, 1),
                            aligned=True,
                        )
                    component_feats[comp_id].append(roi_feat.view(1, -1).cpu())

            sequences_for_patient = []
            for comp_id in sorted(component_feats.keys()):
                feats = torch.cat(component_feats[comp_id], dim=0)
                if len(feats) <= 4:
                    continue
                sequences_for_patient.append(feats)

            if not sequences_for_patient:
                continue

            final_data.append(
                {
                    "pid": f"{pid}_aug{aug_idx}" if dataset_type == "train" else pid,
                    "label": patient_label,
                    "sequences": sequences_for_patient,
                }
            )

    torch.save(final_data, cache_path)
    logging.info(f"Saved extracted {dataset_type} features to {cache_path}")
    return final_data


def run_classification_training(
    fold_identifier: str,
    train_info: List[Dict],
    val_info: List[Dict],
    segmentation_model_path: str,
    roi: float = 0.0,
    w_plus: float = None,
) -> str:
    model_type = get_model_type()
    w_plus = float(config.MIL_W_PLUS if w_plus is None else w_plus)
    logging.info(
        f"========== Starting MIL Training ({fold_identifier}) | method={model_type} | roi={roi} =========="
    )

    segmentation_model, feature_backbone = _load_feature_models(segmentation_model_path)
    train_data = extract_features_from_slices(
        train_info, segmentation_model, feature_backbone, fold_identifier, "train", config.CLASS_AUG_RATIO
    )
    val_data = extract_features_from_slices(
        val_info, segmentation_model, feature_backbone, fold_identifier, "val", {0: 1, 1: 1}
    )

    train_loader = DataLoader(
        PatientComponentDataset(train_data),
        batch_size=config.CLS_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=patient_collate_fn,
    )
    val_loader = DataLoader(
        PatientComponentDataset(val_data),
        batch_size=config.CLS_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=patient_collate_fn,
    )

    model = build_classifier_model(model_type)
    criterion_mean = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.AdamW(model.parameters(), lr=config.CLS_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=20, verbose=True)

    best_val_auc = -1.0
    fold_dir = os.path.join(config.OUTPUT_DIR, str(fold_identifier))
    os.makedirs(fold_dir, exist_ok=True)
    best_model_path = os.path.join(fold_dir, f"best_{model_type}_classifier.pth")

    for epoch in range(config.CLS_EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.CLS_EPOCHS} [{model_type}]")
        for batch in pbar:
            batch = {k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "sequences" not in batch or batch["sequences"].shape[0] == 0:
                continue
            optimizer.zero_grad()
            predictions = model(batch)
            bag_logits = predictions.get("bag_logits")
            instance_logits = predictions.get("instance_logits")

            if model_type == "mil_asym":
                from mil_eval import compute_mil_loss

                loss_patient, _, _ = compute_mil_loss(
                    bag_logits,
                    batch["labels"],
                    batch["component_patient_indices"],
                    w_plus=w_plus,
                    w_minus=float(getattr(config, "MIL_W_MINUS", 1.0)),
                )
            else:
                loss_patient = compute_patient_level_loss(predictions, batch["labels"], criterion_mean, model_type)

            if bag_logits is not None and instance_logits is not None and instance_logits.shape[0] > 0:
                batch_l2_plus, batch_l2_minus = compute_roi_loss(
                    bag_logits,
                    instance_logits,
                    batch["labels"],
                    batch["component_patient_indices"],
                    batch["instance_patient_indices"],
                    batch["instance_component_indices"],
                    criterion_mean,
                )
            else:
                batch_l2_plus, batch_l2_minus = [], []

            loss_roi = (
                torch.sum(torch.stack(batch_l2_plus + batch_l2_minus))
                if (batch_l2_plus or batch_l2_minus)
                else torch.tensor(0.0, device=config.DEVICE)
            )
            total_loss = loss_patient + roi * loss_roi
            if total_loss > 0:
                total_loss.backward()
                optimizer.step()
            pbar.set_postfix(loss=float(total_loss.item()))

        val_metrics = evaluate_classifier(model, val_loader, config.DEVICE, is_mil=True)
        scheduler.step(val_metrics["auc"])
        logging.info(
            f"Epoch {epoch + 1} | Val AUC: {val_metrics['auc']:.4f} | threshold: {val_metrics['optimal_threshold']:.4f}"
        )
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(
                {
                    "model_state_dict": {k: v.clone().detach() for k, v in model.state_dict().items()},
                    "optimal_threshold": val_metrics["optimal_threshold"],
                    "model_type": model_type,
                },
                best_model_path,
            )
            logging.info(f"New best AUC: {best_val_auc:.4f}. Saved to {best_model_path}")

    return best_model_path


def _infer_batch_probs(batch, model, model_type):
    component_patient_indices = batch.get("component_patient_indices")
    dsmil_infer_use_bag = bool(getattr(config, "DSMIL_INFER_USE_BAG_LOGITS", True))
    dsmil_bag_w = float(getattr(config, "DSMIL_LOSS_BAG_WEIGHT", 0.5))
    dsmil_max_w = float(getattr(config, "DSMIL_LOSS_MAX_WEIGHT", 0.5))

    predictions = model(batch)
    patient_logits = predictions.get("patient_logits")
    patient_bag_logits = predictions.get("patient_bag_logits")
    bag_logits = predictions.get("bag_logits")

    if (
        model_type == "dsmil"
        and patient_bag_logits is not None
        and patient_bag_logits.shape[0] == batch["num_patients"]
    ):
        infer_logits = patient_bag_logits if dsmil_infer_use_bag else patient_logits
        if infer_logits is None or infer_logits.shape[0] != batch["num_patients"]:
            infer_logits = dsmil_bag_w * patient_bag_logits + dsmil_max_w * predictions["patient_max_logits"]
        return F.softmax(infer_logits, dim=1)[:, 1]

    if patient_logits is not None and patient_logits.shape[0] == batch["num_patients"]:
        return F.softmax(patient_logits, dim=1)[:, 1]

    if bag_logits is not None and bag_logits.shape[0] > 0:
        bag_probs = F.softmax(bag_logits, dim=1)[:, 1]
        batch_probs = []
        for i in range(batch["num_patients"]):
            mask = component_patient_indices == i
            if torch.any(mask):
                batch_probs.append(torch.max(bag_probs[mask], dim=0).values)
            else:
                batch_probs.append(torch.tensor(0.5, device=bag_probs.device))
        return torch.stack(batch_probs)
    return torch.full((batch["num_patients"],), 0.5, device=config.DEVICE)


def run_classification_testing(
    fold_identifier: str,
    test_info: List[Dict],
    segmentation_model_path: str,
    classifier_model_path: str,
) -> Dict[str, Any]:
    model_type = get_model_type()
    logging.info(f"========== Starting MIL Testing ({fold_identifier}) | method={model_type} ==========")

    segmentation_model, feature_backbone = _load_feature_models(segmentation_model_path)
    test_data = extract_features_from_slices(
        test_info, segmentation_model, feature_backbone, fold_identifier, "test", {0: 1, 1: 1}
    )
    test_loader = DataLoader(
        PatientComponentDataset(test_data),
        batch_size=config.CLS_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=patient_collate_fn,
    )

    model = build_classifier_model(model_type)
    checkpoint = torch.load(classifier_model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimal_threshold = checkpoint["optimal_threshold"]
    model.eval()

    all_targets, all_probs, all_pids = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            pids = batch["pids"]
            targets = batch["labels"]
            batch = {k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "sequences" in batch and batch["sequences"].shape[0] > 0:
                probs = _infer_batch_probs(batch, model, model_type)
            else:
                probs = torch.full((batch["num_patients"],), 0.5, device=config.DEVICE)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_pids.extend(pids)

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    binary_preds = (all_probs >= optimal_threshold).astype(int)

    metrics = {
        "model_type": model_type,
        "auc": roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.5,
        "accuracy": accuracy_score(all_targets, binary_preds),
        "sensitivity": recall_score(all_targets, binary_preds, pos_label=1, zero_division=0),
        "specificity": recall_score(all_targets, binary_preds, pos_label=0, zero_division=0),
        "precision": precision_score(all_targets, binary_preds, zero_division=0),
        "f1_score": f1_score(all_targets, binary_preds, zero_division=0),
        "optimal_threshold": optimal_threshold,
    }

    logging.info("--- Test Results ---")
    for k, v in metrics.items():
        logging.info(f"{k}: {v}")
    logging.info(f"Confusion Matrix:\n{confusion_matrix(all_targets, binary_preds)}")

    for pid, target, prob, pred in zip(all_pids, all_targets, all_probs, binary_preds):
        status = "OK" if pred == target else "WRONG"
        logging.info(f"PID={pid} true={target} prob={prob:.4f} pred={pred} [{status}]")

    return metrics
