import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision.utils import save_image

# MST 模块
from mst.data.datamodules import DataModule
from mst.models.dino import DinoV2ClassifierSlice
from mst.models.resnet import ResNet, ResNetSliceTrans
from mst.utils.roc_curve import plot_roc_curve, cm2acc, cm2x
from mst.models.utils.functions import tensor2image, tensor_cam2image, minmax_norm

# 导入 Dataset 逻辑 (复用 train_kfold_3_1_1.py 中的逻辑)
# 假设你已经把 get_tumor_data_split_3_1_1 和 TumorDataset3D 放在了 mst/data/datasets/dataset_3d_my.py
# 如果没有，请将下面的 get_tumor_dataset 函数粘贴到这里
try:
    from mst.data.datasets.dataset_3d_my import TumorDataset3D, get_tumor_data_split_3_1_1
except ImportError:
    print("Error: 请确保 dataset_3d_my.py 包含 TumorDataset3D 和 get_tumor_data_split_3_1_1")
    exit()

# ==========================================
# 1. 核心配置
# ==========================================
BASE_PATH = "/tmp/nnunet/Dataset507_TumorROIProcessed"
IMG_DIR = os.path.join(BASE_PATH, "imagesTr")

# ==========================================
# 2. Dataset 工厂
# ==========================================
def get_dataset(split, fold=0, num_folds=5):
    # 复用之前定义的划分逻辑
    train_keys, val_keys, test_keys, labels_dict = get_tumor_data_split_3_1_1(fold, num_folds)
    
    # 统一设置深度为 32，适配 batch training
    target_shape = (224, 224, 32)
    
    if split == 'test':
        return TumorDataset3D(
            img_dir=IMG_DIR, keys_list=test_keys, labels_dict=labels_dict,
            target_shape=target_shape, is_train=False
        )
    elif split == 'val':
         return TumorDataset3D(
            img_dir=IMG_DIR, keys_list=val_keys, labels_dict=labels_dict,
            target_shape=target_shape, is_train=False
        )
    else:
        raise ValueError("Predict script only supports test or val split.")

def get_model(name, **kwargs):
    if name == 'ResNet':
        return ResNet
    elif name == 'ResNetSliceTrans':
        return ResNetSliceTrans
    elif name == 'DinoV2ClassifierSlice':
        return DinoV2ClassifierSlice
    else:
        raise ValueError(f"Unknown model: {name}")

# ==========================================
# 3. 预测核心函数 (保留原 MST 逻辑)
# ==========================================
def _pred_trans(model, source, src_key_padding_mask, save_attn=False, use_softmax=True):
    with torch.no_grad():
        pred = model(source, src_key_padding_mask=src_key_padding_mask, save_attn=save_attn)

    if use_softmax: 
        pred = torch.softmax(pred, dim=-1)

    if not save_attn:
        return pred, None, None 

    # Spatial attention (Patch Level)
    weight = model.get_attention_maps()  # [B*D, Heads, HW]
    weight = weight.mean(dim=1) # Mean of heads 
    spatial_shape = torch.tensor(source.shape[3:]) // 14 
    weight = weight.view(1, 1, source.shape[2], *spatial_shape)

    # Slice attention (Slice Level)
    weight_slice = model.get_slice_attention() # [B*D, Heads, 1]
    weight_slice = weight_slice.mean(dim=1) # Mean of heads 
    weight_slice = weight_slice.view(1, 1, -1, 1, 1) * torch.ones_like(source, device=weight.device)
    
    return pred, weight, weight_slice

def run_pred(model, batch, save_attn=False, use_softmax=True):
    source = batch['source'].to(next(model.parameters()).device)
    # src_key_padding_mask 在 MST 的 dataset 实现里如果没提供，默认为 None
    src_key_padding_mask = batch.get('src_key_padding_mask', None) 
    
    if isinstance(model, DinoV2ClassifierSlice) or isinstance(model, ResNetSliceTrans):
        return _pred_trans(model, source, src_key_padding_mask, save_attn, use_softmax)
    else:
        # ResNet baseline logic
        with torch.no_grad():
            pred = model(source)
        if use_softmax: pred = torch.softmax(pred, dim=-1)
        return pred, None, None

# ==========================================
# 4. Main Predict Pipeline
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必须参数: 你的 Checkpoint 路径
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to best checkpoint (.ckpt file)")
    parser.add_argument('--fold', type=int, default=0, help="Fold index used for training this checkpoint")
    parser.add_argument('--model', type=str, default='DinoV2ClassifierSlice')
    
    # 可选参数
    parser.add_argument('--output_dir', default='./results_prediction', type=str)
    parser.add_argument('--get_attention', action='store_true', help='Generate attention maps')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'])
    
    args = parser.parse_args()
    
    # Setup Paths
    path_out = Path(args.output_dir) / f"Fold{args.fold}_{args.split}"
    path_out.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    
    # Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Loading checkpoint: {args.ckpt_path}")

    # ------------ Load Data ----------------
    ds = get_dataset(split=args.split, fold=args.fold, num_folds=5)
    logger.info(f"Dataset Size ({args.split}): {len(ds)}")

    dm = DataModule(ds_test=ds, batch_size=1, num_workers=4) # Batch size 1 for prediction/viz

    # ------------ Load Model ----------------
    # 注意：MST 的 load_best_checkpoint 是加载文件夹，这里我们直接加载 .ckpt 文件更稳妥
    # 使用 Lightning 的 load_from_checkpoint
    model_cls = get_model(args.model)
    
    # 尝试加载
    try:
        model = model_cls.load_from_checkpoint(args.ckpt_path)
    except Exception as e:
        logger.error(f"Failed to load checkpoint via Lightning: {e}")
        logger.info("Trying state_dict load...")
        model = model_cls()
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        
    model.to(device)
    model.eval()

    # ------------ Prediction Loop ----------------
    results = []
    
    # 如果开启 Attention 导出，创建文件夹
    if args.get_attention:
        attn_out_dir = path_out / 'attention'
        attn_out_dir.mkdir(exist_ok=True)

    print("Running prediction...")
    for n, batch in enumerate(tqdm(dm.test_dataloader())):
        source = batch['source']
        target = batch['target']
        uid = batch['uid'][0]
        
        # 运行预测
        # 如果需要 Attention，save_attn=True
        pred, weight, weight_slice = run_pred(model, batch, save_attn=args.get_attention)
        
        # 处理分类结果
        pred_prob = pred[0, 1].item() # 属于 Class 1 (Tumor) 的概率
        pred_cls = torch.argmax(pred, dim=1).item()
        gt = target.item()
        
        results.append({
            'UID': uid,
            'GT': gt,
            'NN': pred_cls,
            'Prob': pred_prob
        })
        
        # ------------ Visualization (Attention Maps) ----------------
        if args.get_attention:
            # 只保存前 20 个或者只保存预测正确的/错误的，防止太慢
            # 这里默认保存全部，如果你数据多可以加 if n > 20: break
            
            # 归一化权重用于可视化
            # weight shape: [1, 1, D, H/14, W/14] -> interpolate to [1, 1, D, H, W]
            weight = F.interpolate(weight, size=source.shape[2:], mode='trilinear')
            
            weight = weight.detach().cpu()
            weight = weight.clip(*np.quantile(weight, [0.995, 0.999])) # 去除极端值
            
            weight_slice = weight_slice.detach().cpu()
            
            # 选择展示切片：展示原图中这几张切片
            # 1. 中间切片
            # 2. Attention 最大的切片
            
            # 计算每张切片的总 Attention
            slice_scores = weight_slice.view(-1).numpy() # [D]
            best_slice_idx = np.argmax(slice_scores)
            
            # 保存原图和叠加图 (Best Slice)
            src_slice = source[0, :, best_slice_idx] # [C, H, W]
            att_slice = weight[0, :, best_slice_idx] # [1, H, W]
            
            # 保存
            save_image(tensor2image(src_slice), attn_out_dir / f'{uid}_src.png', normalize=True)
            
            # 叠加图
            overlay = tensor_cam2image(minmax_norm(src_slice), minmax_norm(att_slice), alpha=0.5)
            save_image(overlay, attn_out_dir / f'{uid}_overlay_best_slice.png', normalize=False)

    # ------------ Metrics Calculation ----------------
    df = pd.DataFrame(results)
    csv_path = path_out / 'predictions.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Predictions saved to {csv_path}")
    
    # 1. Accuracy
    acc = accuracy_score(df['GT'], df['NN'])
    logger.info(f"Accuracy: {acc:.4f}")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(df['GT'], df['NN'])
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # 3. ROC Curve
    fig, axis = plt.subplots(figsize=(6, 6))
    tprs, fprs, auc_val, thrs, opt_idx, _ = plot_roc_curve(df['GT'], df['Prob'], axis)
    fig.savefig(path_out / 'roc_curve.png', dpi=300)
    logger.info(f"AUC: {auc_val:.4f}")
    
    # 4. Confusion Matrix Plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title(f'Confusion Matrix (Acc={acc:.2f})')
    plt.savefig(path_out / 'confusion_matrix.png', dpi=300)
    
    print("Done!")