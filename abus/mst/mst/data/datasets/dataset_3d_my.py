import os
import json
import torch
import torchio as tio
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
import torch.multiprocessing as mp
from typing import Tuple, Union, Optional, Sequence
from typing import Dict, List, Union
from collections import defaultdict
# ==========================================
# 1. 他们的自定义数据增强类 (原样粘贴你的代码)
# ==========================================

from torchio.typing import TypeRangeFloat, TypeTripletInt
from torchio.transforms.transform import TypeMaskingMethod 
from torchio import Subject, Image

BASE_PATH = "/tmp/nnunet/Dataset507_TumorROIProcessed"
IMG_DIR = os.path.join(BASE_PATH, "imagesTr")
LABEL_JSON_PATH = os.path.join(BASE_PATH, "classification_labels.json")
PID_MAP_PATH = os.path.join(BASE_PATH, "pid_to_case_number.json")

class SubjectToTensor(object):
    """Transforms TorchIO Subjects into a Python dict and changes axes order from TorchIO to Torch"""
    def __call__(self, subject: Subject):
        return {key: val.data.swapaxes(1,-1) if isinstance(val, Image) else val  for key,val in subject.items()}

class ImageToTensor(object):
    """Transforms TorchIO Image into a Numpy/Torch Tensor and changes axes order from TorchIO [B, C, W, H, D] to Torch [B, C, D, H, W]"""
    def __call__(self, image: Image):
        return image.data.swapaxes(1,-1)

class ImageOrSubjectToTensor(object):
    """Depending on the input, it will either run SubjectToTensor or ImageToTensor"""
    def __call__(self, input: Union[Image, Subject]):
        if isinstance(input, Subject):
            return {key: val.data.swapaxes(1,-1) if isinstance(val, Image) else val  for key,val in input.items()}
        else:
            return input.data.swapaxes(1,-1)

def parse_per_channel(per_channel, channels):
    if isinstance(per_channel, bool):
        if per_channel == True:
            return [(ch,) for ch in range(channels)]
        else:
            return [tuple(ch for ch in range(channels))] 
    else:
        return per_channel 

class ZNormalization(tio.ZNormalization):
    """Add option 'per_channel' to apply znorm for each channel independently and percentiles to clip values first"""
    def __init__(
        self,
        percentiles: TypeRangeFloat = (0, 100),
        per_channel=True,
        per_slice=False,
        masking_method: TypeMaskingMethod = None,
        **kwargs
    ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.percentiles = percentiles
        self.per_channel = per_channel
        self.per_slice =  per_slice

    def apply_normalization(
        self,
        subject: Subject,
        image_name: str,
        mask: torch.Tensor,
    ) -> None:
        image = subject[image_name]
        per_channel = parse_per_channel(self.per_channel, image.shape[0])
        per_slice = parse_per_channel(self.per_slice, image.shape[-1])

        image.set_data(
            torch.cat([
                torch.cat([
                    self._znorm(image.data[chs,][:,:,:, sl,], mask[chs,][:,:,:, sl,], image_name, image.path)
                for sl in per_slice], dim=-1)
            for chs in per_channel ])
        )
  
    def _znorm(self, image_data, mask, image_name, image_path):
        cutoff = torch.quantile(image_data.masked_select(mask).float(), torch.tensor(self.percentiles)/100.0)
        torch.clamp(image_data, *cutoff.to(image_data.dtype).tolist(), out=image_data)

        standardized = self.znorm(image_data, mask)
        if standardized is None:
            # Fallback if std is 0 (e.g. empty background), avoid crashing
            return torch.zeros_like(image_data) 
        return standardized

class CropOrPad(tio.CropOrPad):
    """CropOrPad. 
     random_center: Random center for crop and pad if no mask is set otherwise only random padding."""
    def __init__(
        self,
        target_shape: Union[int, TypeTripletInt, None] = None,
        padding_mode: Union[str, float] = 0,
        mask_name: Optional[str] = None,
        labels: Optional[Sequence[int]] = None,
        random_center=False,
        **kwargs,
    ):
        super().__init__(
            target_shape=target_shape,
            padding_mode=padding_mode,
            mask_name=mask_name,
            labels=labels,
            **kwargs
        )
        self.random_center = random_center
        
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject.check_consistent_space()
        padding_params, cropping_params = self.compute_crop_or_pad(subject)
        padding_kwargs = {'padding_mode': self.padding_mode}
        if padding_params is not None:
            if self.random_center:
                random_padding_params = []
                for i in range(0, len(padding_params), 2):
                    s = padding_params[i] + padding_params[i + 1]
                    r = np.random.randint(0, s+1)
                    random_padding_params.extend([r, s - r])
                padding_params = random_padding_params
            pad = tio.Pad(padding_params, **padding_kwargs)
            subject = pad(subject)  # type: ignore[assignment]
        if cropping_params is not None:
            crop = tio.Crop(cropping_params)
            subject = crop(subject)  # type: ignore[assignment]
        return subject

# ==========================================
# 2. DataModule (原样保留)
# ==========================================
class DataModule(pl.LightningDataModule):
    def __init__(self, ds_train=None, ds_val=None, ds_test=None, batch_size=1, 
                 batch_size_val=None, batch_size_test=None, num_train_samples=None, 
                 num_workers=4, seed=0, pin_memory=False, weights=None):
        super().__init__()
        self.ds_train = ds_train 
        self.ds_val = ds_val 
        self.ds_test = ds_test 
        self.batch_size = batch_size
        self.batch_size_val = batch_size if batch_size_val is None else batch_size_val 
        self.batch_size_test = batch_size if batch_size_test is None else batch_size_test 
        self.num_train_samples = num_train_samples
        self.num_workers = num_workers
        self.seed = seed 
        self.pin_memory = pin_memory
        self.weights = weights

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_train is not None:
            if self.weights is not None:
                sampler = WeightedRandomSampler(self.weights, num_samples=len(self.weights), generator=generator)
                return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers, 
                                sampler=sampler, generator=generator, drop_last=True, pin_memory=self.pin_memory)
            else:
                return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers, 
                                shuffle=True, generator=generator, drop_last=True, pin_memory=self.pin_memory)
        return None

    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_val is not None:
            return DataLoader(self.ds_val, batch_size=self.batch_size_val, num_workers=self.num_workers, shuffle=False, 
                                generator=generator, drop_last=False, pin_memory=self.pin_memory)
        return None

# ==========================================
# 3. TumorDataset3D (集成他们的增强策略)
# ==========================================
class TumorDataset3D(torch.utils.data.Dataset):
    def __init__(
            self,
            img_dir,
            keys_list,
            labels_dict,
            image_resize=(224, 224, -1), # 长宽 resize，深度保留
            target_shape=(224, 224, 32), # 最终 CropOrPad 到的形状 (含深度)
            is_train=True
        ):
        self.img_dir = Path(img_dir)
        self.keys_list = keys_list
        self.labels_dict = labels_dict
        
        # 严格复刻 DUKE_Dataset3D 中的 transform 逻辑
        if is_train:
            self.transform = tio.Compose([
                tio.Resize(image_resize), # 先 Resize (224, 224, D_original)
                tio.Flip(1), # Just for viewing
                
                # 使用自定义的 CropOrPad (含深度调整)
                # random_center=True 在训练时启用随机裁剪，有助于泛化
                CropOrPad(target_shape, random_center=True, padding_mode='minimum'), 
                
                # 使用自定义的 ZNormalization (带 percentiles 截断)
                ZNormalization(per_channel=True, per_slice=False, 
                               masking_method=lambda x:(x>x.min()) & (x<x.max()), 
                               percentiles=(0.5, 99.5)), 
                
                # 数据增强
                tio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, 0, 90), translation=0, isotropic=True, default_pad_value='minimum'),
                tio.RandomFlip((0,1,2)),
                
                # 转为 Tensor 并换轴
                ImageOrSubjectToTensor() 
            ])
        else:
            # 验证集/测试集 (不进行随机增强，且裁剪中心固定)
            self.transform = tio.Compose([
                tio.Resize(image_resize),
                tio.Flip(1),
                CropOrPad(target_shape, random_center=False, padding_mode='minimum'), # 固定中心裁剪
                ZNormalization(per_channel=True, per_slice=False, 
                               masking_method=lambda x:(x>x.min()) & (x<x.max()), 
                               percentiles=(0.5, 99.5)),
                ImageOrSubjectToTensor()
            ])

    def __len__(self):
        return len(self.keys_list)

    def __getitem__(self, index):
        case_name = self.keys_list[index]
        label = self.labels_dict[case_name]
        
        img_path = self.img_dir / f"{case_name}_0000.nii.gz"
        
        subject = tio.Subject(
            source=tio.ScalarImage(img_path),
            target=label,
            uid=case_name
        )
        
        # 应用增强 (会自动调用 ImageOrSubjectToTensor)
        # 结果是一个 dict，因为我们使用了 Subject
        transformed_subject = self.transform(subject)
        
        # ImageOrSubjectToTensor 已经做了 swapaxes(1, -1) -> [C, D, H, W]
        # 但要注意它返回的是一个 dict: {'uid':..., 'source': tensor, 'target':...}
        img_tensor = transformed_subject['source'] 
        
        # 最终返回
        return {
            'source': img_tensor.float(), # Shape should be [C, D, H, W]
            'target': torch.tensor(label, dtype=torch.long),
            'uid': case_name
        }

def create_patient_folds(patient_labels: Dict[str, int], num_folds: int, seed: int) -> List[List[str]]:
    from sklearn.model_selection import KFold
    patient_ids = np.array(sorted(list(patient_labels.keys())))
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = [patient_ids[test_index].tolist() for _, test_index in kf.split(patient_ids)]
    for i, fold_pids in enumerate(folds):
        print(f"  - Fold {i+1} created with {len(fold_pids)} patients.")
        print(fold_pids)
    return folds

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
# 4. 测试运行
# ==========================================
if __name__ == "__main__":
    # 配置
    base_path = "/tmp/nnunet/Dataset507_TumorROIProcessed"
    img_dir = os.path.join(base_path, "imagesTr")
    json_path = os.path.join(base_path, "classification_labels.json")
    
    with open(json_path, 'r') as f:
        labels_dict = json.load(f)
    keys = list(labels_dict.keys())
    train_keys, val_keys = train_test_split(keys, test_size=0.2, random_state=42)
    
    # 【关键】这里我设置 target_shape 的深度为 32
    # 这意味着无论原始切片多少，最后都会变成 32 (随机裁剪或填充)
    # 这样就能支持 batch_size > 1
    target_depth = 32
    
    ds_train = TumorDataset3D(img_dir, train_keys, labels_dict, target_shape=(224, 224, target_depth), is_train=True)
    ds_val = TumorDataset3D(img_dir, val_keys, labels_dict, target_shape=(224, 224, target_depth), is_train=False)
    
    dm = DataModule(ds_train=ds_train, ds_val=ds_val, batch_size=2, num_workers=0)
    
    loader = dm.train_dataloader()
    print("Start loading batch...")
    batch = next(iter(loader))
    
    # 验证维度
    # ImageOrSubjectToTensor 实际上把 TorchIO 的 [C, W, H, D] 变成了 [C, W, H, D].swapaxes(1,-1)
    # 等等，我们需要仔细检查 swapaxes 的行为。
    # TorchIO 原始 data 是 [C, W, H, D]。
    # swapaxes(1, -1) -> 交换 W 和 D -> [C, D, H, W]。
    # 这是 PyTorch 3D 卷积常用的 [C, D, H, W]。
    
    images = batch['source'] # [B, C, D, H, W]
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {batch['target'].shape}")
    
    # 验证 slices2rgb 兼容性
    # 假设 images 是 [2, 1, 32, 224, 224]
    if images.shape[2] % 3 != 0:
        print(f"注意: 切片数 {images.shape[2]} 不能被3整除，模型内部会有 padding。")
    else:
        print("切片数完美适配。")
