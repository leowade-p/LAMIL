# mil_eval.py - MIL 损失与评估函数
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from mil_model_factory import get_model_type


def compute_mil_loss(component_logits, labels, patient_indices, w_plus=5.0, w_minus=1.0):
    loss_positive = torch.tensor(0.0, device=component_logits.device)
    loss_negative = torch.tensor(0.0, device=component_logits.device)

    if component_logits is None or component_logits.shape[0] == 0:
        return loss_positive + loss_negative, loss_positive, loss_negative

    component_probs_pos = F.softmax(component_logits, dim=1)[:, 1]
    positive_mask = labels == 1
    negative_mask = labels == 0

    if torch.any(positive_mask):
        pos_losses = []
        for i in positive_mask.nonzero().flatten():
            mask_i = patient_indices == i
            if torch.any(mask_i):
                max_prob = component_probs_pos[mask_i].max()
                pos_losses.append(
                    F.binary_cross_entropy(max_prob, torch.ones_like(max_prob), reduction="mean")
                )
        if pos_losses:
            loss_positive = torch.stack(pos_losses).mean()

    if torch.any(negative_mask):
        neg_losses = []
        for i in negative_mask.nonzero().flatten():
            mask_i = patient_indices == i
            if torch.any(mask_i):
                neg_probs = component_probs_pos[mask_i]
                neg_losses.append(
                    F.binary_cross_entropy(neg_probs, torch.zeros_like(neg_probs), reduction="mean")
                )
        if neg_losses:
            loss_negative = torch.stack(neg_losses).mean()

    return w_plus * loss_positive + w_minus * loss_negative, loss_positive, loss_negative


def compute_roi_loss(
    bag_logits,
    instance_logits,
    labels,
    component_patient_indices,
    instance_patient_indices,
    instance_component_indices,
    criterion_mean,
):
    batch_l2_plus, batch_l2_minus = [], []
    if bag_logits is None or instance_logits is None:
        return batch_l2_plus, batch_l2_minus

    for i in range(len(labels)):
        patient_label = labels[i]
        comp_mask = component_patient_indices == i
        inst_mask = instance_patient_indices == i
        if not torch.any(comp_mask):
            continue

        if patient_label == 1:
            patient_component_logits = bag_logits[comp_mask]
            if patient_component_logits.shape[0] > 0:
                _, champion_local_idx = torch.max(patient_component_logits[:, 1], dim=0)
                comp_global_ids = comp_mask.nonzero().flatten()
                if champion_local_idx < len(comp_global_ids):
                    champion_id = comp_global_ids[champion_local_idx]
                    champion_mask = instance_component_indices == champion_id
                    champion_logits = instance_logits[champion_mask]
                    if champion_logits.shape[0] > 0:
                        champion_labels = torch.ones(
                            champion_logits.shape[0], dtype=torch.long, device=instance_logits.device
                        )
                        batch_l2_plus.append(criterion_mean(champion_logits, champion_labels))
        else:
            patient_instance_logits = instance_logits[inst_mask]
            if patient_instance_logits.shape[0] > 0:
                neg_labels = torch.zeros(
                    patient_instance_logits.shape[0], dtype=torch.long, device=instance_logits.device
                )
                batch_l2_minus.append(criterion_mean(patient_instance_logits, neg_labels))

    return batch_l2_plus, batch_l2_minus


def compute_patient_level_loss(predictions, labels, criterion_mean, model_type=None):
    model_type = (model_type or get_model_type()).lower()
    device = labels.device

    patient_logits = predictions.get("patient_logits")
    patient_bag_logits = predictions.get("patient_bag_logits")
    patient_max_logits = predictions.get("patient_max_logits")
    tier1_logits = predictions.get("tier1_logits")
    tier1_patient_indices = predictions.get("tier1_patient_indices")
    clam_instance_loss = predictions.get("clam_instance_loss")
    bag_logits = predictions.get("bag_logits")

    dsmil_bag_w = float(getattr(config, "DSMIL_LOSS_BAG_WEIGHT", 0.5))
    dsmil_max_w = float(getattr(config, "DSMIL_LOSS_MAX_WEIGHT", 0.5))
    dtfd_t1_w = float(getattr(config, "DTFD_LOSS_TIER1_WEIGHT", 1.0))
    dtfd_t2_w = float(getattr(config, "DTFD_LOSS_TIER2_WEIGHT", 1.0))
    clam_bag_w = float(getattr(config, "CLAM_BAG_WEIGHT", 0.7))
    clam_no_inst = bool(getattr(config, "CLAM_NO_INST_CLUSTER", False))

    if (
        model_type == "dsmil"
        and patient_bag_logits is not None
        and patient_max_logits is not None
        and patient_bag_logits.shape[0] == labels.shape[0]
    ):
        loss_bag = criterion_mean(patient_bag_logits, labels)
        loss_max = criterion_mean(patient_max_logits, labels)
        return dsmil_bag_w * loss_bag + dsmil_max_w * loss_max

    if model_type == "dtfd" and patient_logits is not None and patient_logits.shape[0] == labels.shape[0]:
        loss_t2 = criterion_mean(patient_logits, labels)
        if tier1_logits is not None and tier1_patient_indices is not None and tier1_logits.shape[0] > 0:
            loss_t1 = criterion_mean(tier1_logits, labels[tier1_patient_indices])
        else:
            loss_t1 = torch.tensor(0.0, device=device)
        return dtfd_t1_w * loss_t1 + dtfd_t2_w * loss_t2

    if model_type == "clam" and patient_logits is not None and patient_logits.shape[0] == labels.shape[0]:
        bag_loss = criterion_mean(patient_logits, labels)
        if clam_no_inst:
            return bag_loss
        inst_loss = clam_instance_loss if clam_instance_loss is not None else torch.tensor(0.0, device=device)
        return clam_bag_w * bag_loss + (1.0 - clam_bag_w) * inst_loss

    if patient_logits is not None and patient_logits.shape[0] == labels.shape[0]:
        return criterion_mean(patient_logits, labels)

    if bag_logits is not None and bag_logits.shape[0] > 0:
        loss, _, _ = compute_mil_loss(
            bag_logits,
            labels,
            predictions.get("_component_patient_indices"),
            w_plus=float(getattr(config, "MIL_W_PLUS", 5.0)),
            w_minus=float(getattr(config, "MIL_W_MINUS", 1.0)),
        )
        return loss

    return torch.tensor(0.0, device=device)


def evaluate_classifier(model, loader, device, is_mil=False):
    model.eval()
    model_type = get_model_type()
    dsmil_bag_w = float(getattr(config, "DSMIL_LOSS_BAG_WEIGHT", 0.5))
    dsmil_max_w = float(getattr(config, "DSMIL_LOSS_MAX_WEIGHT", 0.5))
    dsmil_infer_use_bag = bool(getattr(config, "DSMIL_INFER_USE_BAG_LOGITS", True))
    dtfd_t1_w = float(getattr(config, "DTFD_LOSS_TIER1_WEIGHT", 1.0))
    dtfd_t2_w = float(getattr(config, "DTFD_LOSS_TIER2_WEIGHT", 1.0))
    clam_bag_w = float(getattr(config, "CLAM_BAG_WEIGHT", 0.7))
    clam_no_inst = bool(getattr(config, "CLAM_NO_INST_CLUSTER", False))

    total_val_loss = 0.0
    all_targets, all_pred_probs = [], []
    bag_criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            targets = batch["labels"]
            component_patient_indices = batch.get("component_patient_indices")
            batch_final_probs = []
            per_patient_losses = []

            if "sequences" in batch and batch["sequences"].shape[0] > 0 and is_mil:
                predictions = model(batch)
                patient_logits = predictions.get("patient_logits")
                patient_bag_logits = predictions.get("patient_bag_logits")
                patient_max_logits = predictions.get("patient_max_logits")
                tier1_logits = predictions.get("tier1_logits")
                tier1_patient_indices = predictions.get("tier1_patient_indices")
                clam_instance_loss = predictions.get("clam_instance_loss")
                bag_logits = predictions.get("bag_logits")

                if (
                    model_type == "clam"
                    and patient_logits is not None
                    and patient_logits.shape[0] == batch["num_patients"]
                ):
                    probs_tensor = F.softmax(patient_logits, dim=1)[:, 1]
                    bag_loss = F.cross_entropy(patient_logits, targets, reduction="mean")
                    if clam_no_inst:
                        total_val_loss += bag_loss.item()
                    else:
                        inst_loss = (
                            clam_instance_loss
                            if clam_instance_loss is not None
                            else torch.tensor(0.0, device=device)
                        )
                        total_val_loss += (clam_bag_w * bag_loss + (1.0 - clam_bag_w) * inst_loss).item()
                    all_targets.extend(targets.cpu().numpy())
                    all_pred_probs.extend(probs_tensor.cpu().numpy())
                    continue

                if (
                    model_type == "dtfd"
                    and patient_logits is not None
                    and patient_logits.shape[0] == batch["num_patients"]
                ):
                    probs_tensor = F.softmax(patient_logits, dim=1)[:, 1]
                    loss_t2 = F.cross_entropy(patient_logits, targets, reduction="mean")
                    if (
                        tier1_logits is not None
                        and tier1_patient_indices is not None
                        and tier1_logits.shape[0] > 0
                    ):
                        loss_t1 = F.cross_entropy(tier1_logits, targets[tier1_patient_indices], reduction="mean")
                    else:
                        loss_t1 = torch.tensor(0.0, device=device)
                    total_val_loss += (dtfd_t1_w * loss_t1 + dtfd_t2_w * loss_t2).item()
                    all_targets.extend(targets.cpu().numpy())
                    all_pred_probs.extend(probs_tensor.cpu().numpy())
                    continue

                if (
                    model_type == "dsmil"
                    and patient_bag_logits is not None
                    and patient_max_logits is not None
                    and patient_bag_logits.shape[0] == batch["num_patients"]
                ):
                    infer_logits = patient_bag_logits if dsmil_infer_use_bag else patient_logits
                    if infer_logits is None or infer_logits.shape[0] != batch["num_patients"]:
                        infer_logits = dsmil_bag_w * patient_bag_logits + dsmil_max_w * patient_max_logits
                    probs_tensor = F.softmax(infer_logits, dim=1)[:, 1]
                    loss_bag = F.cross_entropy(patient_bag_logits, targets, reduction="mean")
                    loss_max = F.cross_entropy(patient_max_logits, targets, reduction="mean")
                    total_val_loss += (dsmil_bag_w * loss_bag + dsmil_max_w * loss_max).item()
                    all_targets.extend(targets.cpu().numpy())
                    all_pred_probs.extend(probs_tensor.cpu().numpy())
                    continue

                if patient_logits is not None and patient_logits.shape[0] == batch["num_patients"]:
                    probs_tensor = F.softmax(patient_logits, dim=1)[:, 1]
                    total_val_loss += F.cross_entropy(patient_logits, targets, reduction="mean").item()
                    all_targets.extend(targets.cpu().numpy())
                    all_pred_probs.extend(probs_tensor.cpu().numpy())
                    continue

                if bag_logits is not None and bag_logits.shape[0] > 0:
                    bag_probs_pos = F.softmax(bag_logits, dim=1)[:, 1]
                    for i in range(batch["num_patients"]):
                        mask = component_patient_indices == i
                        if not torch.any(mask):
                            batch_final_probs.append(torch.tensor(0.5, device=device))
                            continue
                        patient_component_logits = bag_logits[mask]
                        patient_component_probs = bag_probs_pos[mask]
                        max_prob, _ = torch.max(patient_component_probs, dim=0)
                        batch_final_probs.append(max_prob)
                        if targets[i] == 1:
                            _, champion_idx = torch.max(patient_component_logits[:, 1], dim=0)
                            champion_logit = patient_component_logits[champion_idx].unsqueeze(0)
                            per_patient_losses.append(
                                bag_criterion(champion_logit, torch.ones(1, dtype=torch.long, device=device))
                            )
                        else:
                            neg_labels = torch.zeros(
                                patient_component_logits.shape[0], dtype=torch.long, device=device
                            )
                            per_patient_losses.append(bag_criterion(patient_component_logits, neg_labels))
                    probs_tensor = torch.stack(batch_final_probs)
                else:
                    probs_tensor = torch.full((batch["num_patients"],), 0.5, device=device)
            elif batch["num_patients"] > 0:
                probs_tensor = torch.full((batch["num_patients"],), 0.5, device=device)
            else:
                continue

            if per_patient_losses:
                total_val_loss += torch.mean(torch.stack(per_patient_losses)).item()
            all_targets.extend(targets.cpu().numpy())
            all_pred_probs.extend(probs_tensor.cpu().numpy())

    avg_loss = total_val_loss / len(loader) if len(loader) > 0 else 0.0
    metrics = {"loss": avg_loss}

    if len(np.unique(all_targets)) > 1:
        metrics["auc"] = roc_auc_score(all_targets, all_pred_probs)
        f1_scores = []
        thresholds = np.linspace(0, 1, 5001)
        for threshold in thresholds:
            binary_preds = (np.array(all_pred_probs) >= threshold).astype(int)
            f1_scores.append(f1_score(all_targets, binary_preds, zero_division=0))
        optimal_idx = int(np.argmax(f1_scores))
        optimal_threshold = thresholds[optimal_idx]
        max_f1 = f1_scores[optimal_idx]
    else:
        metrics["auc"] = 0.5
        optimal_threshold = 0.5
        max_f1 = 0.0

    metrics["optimal_threshold"] = optimal_threshold
    metrics["max_f1"] = max_f1
    binary_preds = (np.array(all_pred_probs) >= optimal_threshold).astype(int)
    metrics["accuracy"] = accuracy_score(all_targets, binary_preds)
    metrics["sensitivity"] = recall_score(all_targets, binary_preds, zero_division=0)
    metrics["precision"] = precision_score(all_targets, binary_preds, zero_division=0)
    metrics["f1_score"] = f1_score(all_targets, binary_preds, zero_division=0)
    try:
        tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds).ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    except ValueError:
        metrics["specificity"] = 0
    return metrics
