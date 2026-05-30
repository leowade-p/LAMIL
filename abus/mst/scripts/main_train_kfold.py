import argparse
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import torch 
import wandb 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# 导入 MST 模块
from mst.data.datamodules import DataModule
from mst.models.resnet import ResNet, ResNetSliceTrans
from mst.models.dino import DinoV2ClassifierSlice

# 导入 Dataset 类
try:
    from mst.data.datasets.dataset_3d_my import TumorDataset3D
except ImportError:
    print("Error: 找不到 TumorDataset3D，请确认文件位置。")
    exit()

# ==========================================
# 1. 配置路径
# ==========================================
BASE_PATH = "/tmp/nnunet/Dataset507_TumorROIProcessed"
IMG_DIR = os.path.join(BASE_PATH, "imagesTr")
LABEL_JSON_PATH = os.path.join(BASE_PATH, "classification_labels.json")
PID_MAP_PATH = os.path.join(BASE_PATH, "pid_to_case_number.json")

# ==========================================
# 2. 划分函数 (生成 K 个 Folds)
# ==========================================
def create_patient_folds(patient_labels: Dict[str, int], num_folds: int, seed: int) -> List[List[str]]:
    from sklearn.model_selection import KFold
    patient_ids = np.array(sorted(list(patient_labels.keys())))
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = [patient_ids[test_index].tolist() for _, test_index in kf.split(patient_ids)]
    for i, fold_pids in enumerate(folds):
        print(f"  - Fold {i+1} created with {len(fold_pids)} patients.")
        print(fold_pids)
    return folds

# ==========================================
# 3. 核心逻辑：3 Train / 1 Val / 1 Test + 5倍扩增
# ==========================================
def get_tumor_data_split_3_1_1(fold_index, num_folds=5, seed=42):
    """
    实现逻辑:
    Test Fold = fold_index
    Val Fold  = (fold_index + 1) % num_folds
    Train Folds = 剩下的 3 折
    Train Set = Train Folds * 5 (5倍扩增)
    """
    
    # 1. 加载映射和标签
    with open(LABEL_JSON_PATH, 'r') as f:
        labels_dict = json.load(f) # Filename -> Label
        
    with open(PID_MAP_PATH, 'r') as f:
        pid_map = json.load(f) # PID -> CaseNumber (int)

    # 2. 建立 PID -> Filename 和 PID -> Label 映射
    pid_to_filename = {}
    patient_labels_for_split = {}
    
    for pid_raw, case_num in pid_map.items():
        # 格式化文件名: 0 -> "TumorROIProcessed_000"
        filename = f"TumorROIProcessed_{case_num:03d}"
        
        if filename in labels_dict:
            pid_to_filename[pid_raw] = filename
            patient_labels_for_split[pid_raw] = labels_dict[filename]
    
    # 3. 生成 5 折
    # list of lists, e.g., [[pid1, pid2], [pid3, pid4]...]
    all_folds_pids = create_patient_folds(patient_labels_for_split, num_folds=num_folds, seed=seed)
    
    # 4. 确定索引
    test_fold_idx = fold_index
    val_fold_idx = (fold_index + 1) % num_folds
    # 剩下的就是训练集索引
    train_fold_indices = [j for j in range(num_folds) if j != test_fold_idx and j != val_fold_idx]
    
    print(f"#################### FOLD {fold_index}/{num_folds} ####################")
    print(f"Indices -> Test: {test_fold_idx}, Val: {val_fold_idx}, Train: {train_fold_indices}")

    # 5. 提取 PID
    # Test PIDs
    test_pids = all_folds_pids[test_fold_idx]
    
    # Val PIDs
    val_pids = all_folds_pids[val_fold_idx]
    
    # Train PIDs (Base)
    train_pids_base = []
    for idx in train_fold_indices:
        train_pids_base.extend(all_folds_pids[idx])
        
    # === 【关键】实现训练集 5 倍扩增 ===
    # 列表乘法: ['a', 'b'] * 2 = ['a', 'b', 'a', 'b']
    train_pids = train_pids_base * 5
    
    # 6. PID 转 Filename
    # 辅助函数：安全转换
    def pids_to_keys(pids):
        return [pid_to_filename[pid] for pid in pids if pid in pid_to_filename]

    train_keys = pids_to_keys(train_pids)
    val_keys = pids_to_keys(val_pids)
    test_keys = pids_to_keys(test_pids)
    
    print(f"Patient Counts: Train(Base)={len(train_pids_base)}, Train(Aug)={len(train_pids)}, Val={len(val_pids)}, Test={len(test_pids)}")
    print(f"File Counts:    Train(Aug)={len(train_keys)}, Val={len(val_keys)}, Test={len(test_keys)}")

    return train_keys, val_keys, test_keys, labels_dict

# ==========================================
# 4. Dataset Factory
# ==========================================
def get_dataset(name, split, fold=0, num_folds=5, **kwargs):
    if name == 'TUMOR':
        # 获取切分后的 keys
        train_keys, val_keys, test_keys, labels_dict = get_tumor_data_split_3_1_1(fold, num_folds)
        
        # 统一设置深度为 32，适配 batch training
        target_shape = (224, 224, 32)
        
        if split == 'train':
            return TumorDataset3D(
                img_dir=IMG_DIR, keys_list=train_keys, labels_dict=labels_dict,
                target_shape=target_shape, is_train=True
            )
        elif split == 'val':
            return TumorDataset3D(
                img_dir=IMG_DIR, keys_list=val_keys, labels_dict=labels_dict,
                target_shape=target_shape, is_train=False
            )
        elif split == 'test':
            return TumorDataset3D(
                img_dir=IMG_DIR, keys_list=test_keys, labels_dict=labels_dict,
                target_shape=target_shape, is_train=False
            )
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_model(name, **kwargs):
    if name == 'ResNet':
        return ResNet(in_ch=1, out_ch=2, spatial_dims=3, **kwargs)
    elif name == 'DinoV2ClassifierSlice':
        return DinoV2ClassifierSlice(in_ch=3, out_ch=2, spatial_dims=2, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    import time
    start_time = time.time() # 获取开始时间戳


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TUMOR', choices=['TUMOR'])
    parser.add_argument('--model', type=str, default='DinoV2ClassifierSlice')
    parser.add_argument('--path_root_output', type=str, default='./runs')
    parser.add_argument('--fold', type=int, default=0, help="Current fold index (0-4)")
    
    args = parser.parse_args()

    # 固定 num_folds 为 5 (3+1+1)
    NUM_FOLDS = 5
    
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path(args.path_root_output) / args.dataset / f'{args.model}_Fold{args.fold}_{current_time}'
    path_run_dir.mkdir(parents=True, exist_ok=True)
    
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    # ------------ Load Data (3 sets) ----------------
    ds_train = get_dataset('TUMOR', split='train', fold=args.fold, num_folds=NUM_FOLDS)
    ds_val   = get_dataset('TUMOR', split='val',   fold=args.fold, num_folds=NUM_FOLDS)
    ds_test  = get_dataset('TUMOR', split='test',  fold=args.fold, num_folds=NUM_FOLDS)
    
    batch_size = 16 # 显存允许的话可以调大
    print("dataloader构建好")
    # ------------ Weights Calculation (针对扩增后的 Train) ------------
    # 注意：train_keys 已经是扩增了 5 倍的，所以这里的权重计算会自动适配这个长度
    train_labels = [ds_train.labels_dict[k] for k in ds_train.keys_list]
    train_labels_series = pd.Series(train_labels)
    class_counts = train_labels_series.value_counts()
    
    # 计算权重
    class_weights = 0.5 / class_counts
    weights = train_labels_series.map(lambda x: class_weights[x]).values
    weights = torch.tensor(weights, dtype=torch.double)

    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test, # 传入 Test Set
        batch_size=batch_size, 
        pin_memory=True,
        weights=weights,
        num_workers=8
    )
    print("模型构建好")
    # ------------ Initialize Model ------------
    model = get_model(args.model)
    checkpoint = torch.load('/home/huyiding/pengdie/mst/MST_DUKE.ckpt', map_location='cpu')

    # 3. 提取 state_dict 并加载
    # PyTorch Lightning 的 checkpoint 通常把权重保存在 'state_dict' 键下
    state_dict = checkpoint['state_dict']

    # 有时候 Lightning 会给参数名加前缀 "model." 或 "net."，如果匹配不上需要去除前缀
    # 这里的代码会自动处理简单的加载
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("直接加载失败，尝试去除 key 的前缀...")
        # 这是一个常见的修复 key 不匹配的逻辑
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("model.", "") # 根据实际情况调整，比如可能是 "net."
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    print("模型权重加载成功")

    logger = WandbLogger(
        project=f'Classifier_{args.dataset}', 
        name=f"Fold{args.fold}_3Train_1Val_1Test", 
        log_model=False
    )
    
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor="val/AUC_ROC",
        save_last=True,
        save_top_k=1,
        mode="max",
        filename=f'fold{args.fold}-best-{{epoch:02d}}-{{val/AUC_ROC:.4f}}'
    )
    
    trainer = Trainer(
        accelerator=accelerator,
        precision='16-mixed',
        default_root_dir=str(path_run_dir),
        callbacks=[
            checkpointing, 
            LearningRateMonitor(logging_interval='step'), 
            EarlyStopping(monitor="val/AUC_ROC", patience=30, mode="max")
        ],
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        max_epochs=200,
        logger=logger,
        num_sanity_val_steps=0
    )
   
    # 1. Training & Validation
    print(f"Starting Training...")
    trainer.fit(model, datamodule=dm)

    # 2. Testing (使用最佳模型)
    if checkpointing.best_model_path:
        print(f"Training finished. Loading best model: {checkpointing.best_model_path}")
        print("Starting Testing...")
        # 自动加载最佳权重并在 ds_test 上运行评估
        trainer.test(model, datamodule=dm, ckpt_path='best')
    else:
        print("No best model found (maybe early stopping triggered too soon?), testing with last weights.")
        trainer.test(model, datamodule=dm)
    end_time = time.time() # 获取结束时间戳
    run_time = end_time - start_time # 计算运行时间
    print("程序的运行时间为：", run_time, "秒")
    wandb.finish(quiet=True)