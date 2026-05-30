# scripts/train.py - 3D ResNet (R3D-18) CBBCT baseline

import scripts._bootstrap  # noqa: F401

import logging
import os
import time
from typing import Dict, List

import albumentations as A
import config
import cv2
import numpy as np
import torch
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from dataloader.dataset import ALL_PATIENTS_LABELS, TumorDataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

TARGET_SIZE = config.TARGET_SIZE
TEMPORAL_SIZE = config.TEMPORAL_SIZE
RANDOM_STATE = config.RANDOM_STATE
K_FOLDS = config.K_FOLDS
KINETICS_MEAN = config.KINETICS_MEAN
KINETICS_STD = config.KINETICS_STD
device = config.DEVICE
print(f"Device being used: {device}")

def evaluate_classifier(model, loader, criterion, device, is_test_set=False, fixed_threshold=None):
    model.eval()
    total_loss = 0.0
    all_targets, all_pred_probs = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)
            
            # 我们需要的是类别1的概率
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_targets.extend(targets.cpu().numpy())
            all_pred_probs.extend(probs.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    metrics = {'loss': avg_loss}
    all_targets = np.array(all_targets)
    all_pred_probs = np.array(all_pred_probs)

    # 确保标签是多样的，以计算AUC
    if len(np.unique(all_targets)) > 1:
        metrics['auc'] = roc_auc_score(all_targets, all_pred_probs)
        
        # 寻找最佳阈值 (仅在验证集上进行)
        if not is_test_set:
            thresholds = np.linspace(0, 1, 5001)
            f1_scores = [f1_score(all_targets, (all_pred_probs >= t).astype(int), zero_division=0) for t in thresholds]
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = fixed_threshold
    else:
        # 如果所有样本都属于同一类别
        metrics['auc'] = 0.5
        optimal_threshold = fixed_threshold if is_test_set else 0.5

    metrics['optimal_threshold'] = optimal_threshold
    binary_preds = (all_pred_probs >= optimal_threshold).astype(int)
    
    metrics['accuracy'] = accuracy_score(all_targets, binary_preds)
    metrics['sensitivity'] = recall_score(all_targets, binary_preds, pos_label=1, zero_division=0)
    metrics['precision'] = precision_score(all_targets, binary_preds, pos_label=1, zero_division=0)
    metrics['f1_score'] = f1_score(all_targets, binary_preds, pos_label=1, zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    except ValueError: # 如果只有一个类别，ravel会失败
        metrics['specificity'] = 0
        
    return metrics
# --- Helper Functions ---
def create_patient_folds(patient_labels: Dict[str, int], num_folds: int, seed: int) -> List[List[str]]:
    """
    Splits the dataset at the patient-level into K non-overlapping folds.
    """
    from sklearn.model_selection import KFold
    
    logging.info(f"Creating {num_folds} folds for {len(patient_labels)} patients...")
    patient_ids = np.array(sorted(list(patient_labels.keys())))
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    folds = []
    for _, test_index in kf.split(patient_ids):
        fold_pids = patient_ids[test_index].tolist()
        folds.append(fold_pids)
        
    for i, fold_pids in enumerate(folds):
        logging.info(f"  - Fold {i+1} created with {len(fold_pids)} patients.")
        logging.info(fold_pids)
        
    return folds

# --- Model Definition ---
class Custom3DResNet(nn.Module):
    def __init__(self, num_classes):
        super(Custom3DResNet, self).__init__()
        self.resnet3d = models.video.r3d_18(weights=models.video.R3D_18_Weights.DEFAULT)
        num_features = self.resnet3d.fc.in_features
        self.resnet3d.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet3d(x)

# --- Main Cross-Validation Runner ---
def run_cross_validation(num_classes, directory, num_epochs=100):
    total_start_time = time.time()

    # --- 1. Define Data Augmentations (once) ---
    transform_train = A.Compose([
        A.Rotate(limit=30, p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomScale(scale_limit=0.2, p=0.2, interpolation=cv2.INTER_LINEAR),
        A.RandomGamma(gamma_limit=(70, 150), p=0.1),
        A.LongestMaxSize(max_size=TARGET_SIZE),
        A.PadIfNeeded(min_height=TARGET_SIZE, min_width=TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.CenterCrop(height=TARGET_SIZE, width=TARGET_SIZE),
        A.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD ),
        ToTensorV2(),
    ])
    transform_val_test = A.Compose([
        A.LongestMaxSize(max_size=TARGET_SIZE),
        A.PadIfNeeded(min_height=TARGET_SIZE, min_width=TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.CenterCrop(height=TARGET_SIZE, width=TARGET_SIZE),
        A.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD ),
        ToTensorV2(),
    ])

    # --- 2. Create Patient Folds (once) ---
    all_patient_folds = create_patient_folds(ALL_PATIENTS_LABELS, K_FOLDS, RANDOM_STATE)
    
    test_results_summary = []
    # --- 3. Main Cross-Validation Loop ---
    for i in range(K_FOLDS):
        fold_num = i + 1
        
        # Add fold number to logger for clarity
        extra_args = {'fold': fold_num}
        logger = logging.getLogger()

        logger.info(f"#################### FOLD {fold_num}/{K_FOLDS} ####################", extra=extra_args)

        # 3.1. Assign folds to Train, Val, Test based on your logic
        test_fold_index = i
        val_fold_index = (i + 1) % K_FOLDS # The next fold (wraps around for the last iteration)
        train_fold_indices = [j for j in range(K_FOLDS) if j != test_fold_index and j != val_fold_index]
        
        train_pids = [pid for idx in train_fold_indices for pid in all_patient_folds[idx]]
        val_pids = all_patient_folds[val_fold_index]
        test_pids = all_patient_folds[test_fold_index]

        logger.info(f"Train Folds: {[x+1 for x in train_fold_indices]} | Val Fold: {val_fold_index+1} | Test Fold: {test_fold_index+1}", extra=extra_args)
        logger.info(f"Patient Counts: Train={len(train_pids)}, Val={len(val_pids)}, Test={len(test_pids)}", extra=extra_args)
        
        # 3.2. Create Datasets and DataLoaders for the current fold
        train_dataset = TumorDataset(directory, patient_ids=train_pids, albumentations_transform=transform_train, temporal_size=TEMPORAL_SIZE, mode='train')
        val_dataset   = TumorDataset(directory, patient_ids=val_pids,   albumentations_transform=transform_val_test, temporal_size=TEMPORAL_SIZE, mode='val')
        test_dataset  = TumorDataset(directory, patient_ids=test_pids,  albumentations_transform=transform_val_test, temporal_size=TEMPORAL_SIZE, mode='test')

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        # 3.3. Initialize Model and Optimizer for the current fold
        model = Custom3DResNet(num_classes=num_classes)
        # model = nn.DataParallel(model, device_ids=[0, 1, 4, 5, 6, 7])
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        best_val_acc = 0.0
        best_model_path = f"best_model_fold_{fold_num}.pth"

        # 3.4. Training & Validation Loop for the current fold
        # 3.4. Training & Validation Loop for the current fold
        for epoch in range(num_epochs):
            logger.info(f"--- Epoch {epoch + 1}/{num_epochs} ---", extra=extra_args)

            # -------------------- TRAINING PHASE --------------------
            model.train()
            running_loss_train = 0.0
            running_corrects_train = 0

            for inputs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 前向传播 (在 set_grad_enabled(True) 上下文中)
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播和优化
                    loss.backward()
                    optimizer.step()

                # 统计损失和准确率
                running_loss_train += loss.item() * inputs.size(0)
                running_corrects_train += torch.sum(preds == labels.data)

            epoch_loss_train = running_loss_train / len(train_dataset)
            epoch_acc_train = running_corrects_train.double() / len(train_dataset)
            logger.info(f"Train Loss: {epoch_loss_train:.4f} Acc: {epoch_acc_train:.4f}", extra=extra_args)

            # -------------------- VALIDATION PHASE --------------------
            val_metrics = evaluate_classifier(model, val_loader, criterion, device, is_test_set=False)
            log_str = "Validation Metrics: "
            for k, v in val_metrics.items():
                log_str += f"{k}: {v:.4f} | "
            logger.info(log_str, extra=extra_args)
            
            # 以 AUC 作为选择最佳模型的标准 (也可以改为 'f1_score')
            current_metric = val_metrics['auc']
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_val_threshold = val_metrics['optimal_threshold']
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved with AUC: {best_val_metric:.4f} and Threshold: {best_val_threshold:.4f}", extra=extra_args)

            if scheduler: scheduler.step()


        # -------------------- TESTING PHASE (for the current fold) --------------------
        logger.info(f"--- Evaluating best model of Fold {fold_num} on its test set ---", extra=extra_args)
        
        # 加载本折中验证集上表现最好的模型
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        test_metrics = evaluate_classifier(model, test_loader, criterion, device, 
                                           is_test_set=True, fixed_threshold=best_val_threshold)
        
        log_str = f"****** Fold {fold_num} Test Metrics (Thresh={best_val_threshold:.4f}) ******\n"
        for k, v in test_metrics.items():
            log_str += f"  - {k.capitalize()}: {v:.4f}\n"
        logger.info(log_str, extra=extra_args)
        
        test_results_summary.append(test_metrics)

    # --- 4. Final Summary ---
    print("\n" + "="*50)
    print("CROSS-VALIDATION SUMMARY")
    print("="*50)
    all_accs = [res["accuracy"] for res in test_results_summary]
    for fold_idx, res in enumerate(test_results_summary, start=1):
        print(f"  - Fold {fold_idx} Test Accuracy: {res['accuracy']:.4f}")
    
    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    print(f"\nAverage Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    total_time_elapsed = time.time() - total_start_time
    print(f'Total process complete in {total_time_elapsed // 3600:.0f}h {(total_time_elapsed % 3600) // 60:.0f}m {total_time_elapsed % 60:.0f}s')

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    run_cross_validation(
        num_classes=config.NUM_CLASSES,
        directory=config.DATA_ROOT,
        num_epochs=config.NUM_EPOCHS,
    )


if __name__ == "__main__":
    main()