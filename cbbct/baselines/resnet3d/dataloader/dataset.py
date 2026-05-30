# dataset.py

import os
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import itk
from PIL import Image
import re # 导入正则表达式模块

# 1. 标签字典 (保持不变)
ALL_PATIENTS_LABELS =  {'2195561.nii': 1, '1918325.nii': 1, '1916513.nii': 1, '1852423.nii': 1, '1917881.nii': 1,
     '1857249.nii': 1, '2120967.nii': 1, '1865172.nii': 1, '1913371.nii': 1, '1926461.nii': 1, 
     '1925578.nii': 1, '2185498.nii': 1, '1862468.nii': 1, '1921315.nii': 1, '1854065.nii': 1, 
     '2176703.nii': 1, '2121132.nii': 1, '1918223.nii': 1, '1850916.nii': 1, '2189875.nii': 1, 
     '1924172.nii': 1, '1936357.nii': 1, '2192392.nii': 1, '2054533.nii': 1, '1933813.nii': 1, 
     '2191704.nii': 1, '1860366.nii': 1, '1914739.nii': 1, '1850947.nii': 1, '1926530.nii': 1, 
     '2149937.nii': 1, '2175352.nii': 1, '1859252.nii': 1, '1876395.nii': 1, '2189476.nii': 1, 
     '1910532.nii': 1, '2146979.nii': 1, '1926016.nii': 1, '2120982.nii': 1, '2173278.nii': 1, 
     '1927130.nii': 1, '1872957.nii': 1, '2173219.nii': 1, '2142355.nii': 1, '1917842.nii': 1, 
     '2021293.nii': 1, '1927747.nii': 1, '2186670.nii': 1, '1918272.nii': 1, '2174289.nii': 1, 
     '2191237.nii': 1, '2192966.nii': 1, '1870372.nii': 1, '1912864.nii': 1, '2101138.nii': 1, 
     '1853371.nii': 1, '2092015.nii': 1, '1879483.nii': 1, '1900454.nii': 1, '2194012.nii': 1, 
     '2173398.nii': 1, '1901411.nii': 1, '1916755.nii': 1, '1918233.nii': 1, '2188185.nii': 1,
     '1927047.nii': 1, '1917103.nii': 1, '1867143.nii': 1, '1916422.nii': 1, '1922289.nii': 1, 
     '1922706.nii': 1, '1859176.nii': 1, '2130175.nii': 1, '2196119.nii': 1, '1847203.nii': 1, 
     '2110127.nii': 1, '2132202.nii': 1, '1927553.nii': 1, '2189573.nii': 1, '2086306.nii': 1, 
     '1914670.nii': 1, '1912351.nii': 1, '2145811.nii': 1, '2124977.nii': 1, '1925054.nii': 1, 
     '1918234.nii': 1, '1860410.nii': 1, '2185872.nii': 1, '2154484.nii': 1, '2141440.nii': 1, 
     '2701357.nii': 0, '2336756.nii': 0, '2093629.nii': 0, '1915238.nii': 0, '2699066.nii': 0, 
     '2701358.nii': 0, '1924774.nii': 0, '1921022.nii': 0, '2027345.nii': 0, '1958973.nii': 0, 
     '1915209.nii': 0, '2698551.nii': 0, '1942564.nii': 0, '2561719.nii': 0, '1851144.nii': 0, 
     '1931525.nii': 0, '2597789.nii': 0, '2945516.nii': 0, '1916092.nii': 0, '1926843.nii': 0, 
     '2701177.nii': 0, '1953114.nii': 0, '2577307.nii': 0, '2061898.nii': 0, '2639834.nii': 0, 
     '2130364.nii': 0, '2648358.nii': 0, '2110802.nii': 0, '3138995.nii': 0, '1132045.nii': 0, 
     '3237293.nii': 0, '2042416.nii': 0, '1915203.nii': 0, '2125006.nii': 0, '2696318.nii': 0, 
     '879720.nii': 0, '1925620.nii': 0, '2336755.nii': 0, '2578670.nii': 0, '2013797.nii': 0, 
     '1926144.nii': 0, '879721.nii': 0, '1882547.nii': 0, '1935102.nii': 0, '1378023.nii': 0, 
     '1851152.nii': 0, '2090498.nii': 0, '1915205.nii': 0, '2125005.nii': 0, '1879428.nii': 0, 
     '2110801.nii': 0, '2597790.nii': 0, '1930335.nii': 0, '2796911.nii': 0, '1931526.nii': 0, 
     '2770800.nii': 0, '1915201.nii': 0, '2951334.nii': 0, '2582916.nii': 0, '1915207.nii': 0,
     '1922914.nii': 0, '2027346.nii': 0, '1925759.nii': 0, '2639836.nii': 0, '1925991.nii': 0, 
     '1401139.nii': 0, '1270467.nii': 0, '2577306.nii': 0, '1865567.nii': 0, '1927969.nii': 0, 
     '1915206.nii': 0, '2093628.nii': 0, '2110800.nii': 0, '1915202.nii': 0, '2027348.nii': 0, 
     '1953115.nii': 0, '2027347.nii': 0, '2770801.nii': 0, '1958972.nii': 0, '2639835.nii': 0, 
     '2122585.nii': 0, '1853634.nii': 0, '1945008.nii': 0, '1850703.nii': 0, '1915208.nii': 0, 
     '2141374.nii': 0, '2639837.nii': 0, '2100216.nii': 0, '1880821.nii': 0, '2125004.nii': 0, 
     '2785765.nii': 0, '1921021.nii': 0, '1864666.nii': 0, '2181815.nii': 1, '1920578.nii': 1, 
     '2188198.nii': 1, '1909134.nii': 1, '1913193.nii': 1, '1912052.nii': 1, '1944853.nii': 1, 
     '1953833.nii': 1, '1929233.nii': 1, '1906015.nii': 1, '2153776.nii': 1, '1947628.nii': 1, 
     '1927558.nii': 1, '1928694.nii': 1, '2165239.nii': 1, '1991716.nii': 0, '1865568.nii': 0, 
     '2760906.nii': 0, '2336754.nii': 0, '2728474.nii': 0, '2336753.nii': 0, '1132044.nii': 0, 
     '1879429.nii': 0, '2736060.nii': 0, '238876.nii': 0, '1915204.nii': 0, '1850473.nii': 0, 
     '2701356.nii': 0, '2593021.nii': 0, '1931527.nii': 0}

# 2. 窗宽窗位变换函数 (保持不变)
def normalizatingImage(img, fc, fw):
    fmin = (2.0 * fc - fw) / 2 + 0.5
    fmax = (2.0 * fc + fw) / 2 + 0.5
    img = np.clip(img, fmin, fmax)
    img = (img - fmin) / (fmax - fmin) * 255
    img = Image.fromarray(img)
    if img.mode == "F":
        img = img.convert('RGB')
    return img

# 3. 最终修订的 Dataset 类
# class TumorDataset(Dataset):
#     def __init__(self, directory, spatial_transform=None, temporal_transform=None, mode='train'):
#         self.directory = Path(directory)
#         self.spatial_transform = spatial_transform
#         self.temporal_transform = temporal_transform
#         self.mode = mode
        
#         # --- [全新] 数据加载逻辑 ---
#         # 1. 加载 PID -> Case ID (唯一索引) 的映射
#         pid_map_path = self.directory / 'pid_map_total.json'
#         with open(pid_map_path, 'r') as f:
#             pid_to_case_id_map = json.load(f)
        
#         # 2. 创建一个反向映射: Case ID (唯一索引) -> PID
#         case_id_to_pid_map = {str(v): k for k, v in pid_to_case_id_map.items()}

#         # 3. 扫描 imagesTr 文件夹并构建文件和标签列表
#         images_dir = self.directory / 'imagesTr'
#         self.fnames = []
#         self.label_array = []

#         for fname in sorted(images_dir.glob('*.nii.gz')):
#             # 从文件名 'TumorTotal_001_0000.nii.gz' 中用正则表达式提取 Case ID '001'
#             match = re.search(r'_(\d{3})_', fname.name)
#             if not match:
#                 print(f"警告: 无法从文件名 {fname.name} 中解析Case ID，跳过。")
#                 continue
            
#             case_id_str = str(int(match.group(1))) # 转换为 '1' 而不是 '001'

#             # 4. 链接 Case ID -> PID -> Label
#             if case_id_str in case_id_to_pid_map:
#                 pid = case_id_to_pid_map[case_id_str]
#                 pid_with_suffix = pid + '.nii'
                
#                 if pid_with_suffix in ALL_PATIENTS_LABELS:
#                     self.fnames.append(fname)
#                     self.label_array.append(ALL_PATIENTS_LABELS[pid_with_suffix])
#                 else:
#                     print(f"警告: 找到了PID {pid}，但在标签字典中找不到，跳过文件 {fname.name}")
#             else:
#                 print(f"警告: 找到了Case ID {case_id_str}，但在pid_map中找不到对应PID，跳过文件 {fname.name}")

#         # --- [新功能] 训练集5倍扩增 ---
#         if self.mode == 'train':
#             self.fnames = self.fnames * 5
#             self.label_array = self.label_array * 5
        
#         self.label_array = np.array(self.label_array, dtype=np.int64)
#         print(f"模式: '{self.mode}', 成功加载 {len(self.fnames)} 个样本。")
#         # --------------------

#     def __len__(self):
#         return len(self.fnames)

#     def __getitem__(self, index):
#         fname = self.fnames[index]
#         label = self.label_array[index]

#         itk_image = itk.imread(str(fname), itk.F)
#         # 修正: ITK 图像的维度顺序是 (W, H, D)，所以深度是第2个索引
#         frame_count = itk_image.GetLargestPossibleRegion().GetSize()[2] 

#         # 1. 时间变换
#         frame_indices = list(range(frame_count))
#         if self.temporal_transform is not None:
#             frame_indices = self.temporal_transform(frame_indices)

#         # 2. 空间变换
#         if self.spatial_transform is not None:
#             self.spatial_transform.randomize_parameters()

#         clip = []
#         numpy_image = itk.GetArrayFromImage(itk_image)

#         for i in frame_indices:
#             # 从Numpy数组中获取单帧数据
#             # 此时 frame 是一个 (H, W) 的数组，值域 [0, 255]
#             frame = numpy_image[i, :, :]

#             # 1. 将Numpy数组转换为 'L' 模式 (8-bit灰度) 的PIL Image
#             pil_img_gray = Image.fromarray(np.uint8(frame)).convert('L')

#             # 2. 通过复制灰度通道来创建三通道RGB图像
#             pil_img = Image.merge('RGB', [pil_img_gray, pil_img_gray, pil_img_gray])
            
#             # 3. 应用空间变换 (裁剪、翻转、ToTensor、Normalize等)
#             if self.spatial_transform is not None:
#                 pil_img = self.spatial_transform(pil_img)
            
#             clip.append(pil_img)

        
#         # 3. 堆叠成最终的Tensor
#         if isinstance(clip[0], torch.Tensor):
#              clip = torch.stack(clip, dim=1)
#         else: # 如果空间变换没有包含ToTensor，我们需要自己转
#              clip_tensors = [torch.from_numpy(np.array(img).transpose(2, 0, 1)) for img in clip]
#              clip = torch.stack(clip_tensors, dim=1)


#         return clip, label

# dataset.py

import os
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import itk
from PIL import Image
import re
import albumentations as A
from albumentations.pytorch import ToTensorV2


# 最终修订的 Dataset 类 (Albumentations 版本)
class TumorDataset(Dataset):
    def __init__(self, directory, patient_ids=None, albumentations_transform=None, temporal_size=160, mode='train'):
        self.directory = Path(directory)
        self.patient_ids = patient_ids # <-- 新增参数
        self.albumentations_transform = albumentations_transform
        self.temporal_size = temporal_size
        self.mode = mode
        
        # --- [修改] 数据加载逻辑 ---
        pid_map_path = self.directory / 'pid_map_total.json'
        with open(pid_map_path, 'r') as f:
            pid_to_case_id_map = json.load(f)
        case_id_to_pid_map = {str(v): k for k, v in pid_to_case_id_map.items()}

        images_dir = self.directory / 'imagesTr'
        self.fnames = []
        self.label_array = []
        
        # 如果没有提供病人ID列表，则加载所有病人
        if self.patient_ids is None:
            self.patient_ids = list(ALL_PATIENTS_LABELS.keys())
        
        # 根据 patient_ids 列表来构建数据集
        # 创建一个 PID -> Case ID 的反向映射以便快速查找
        pid_to_case_id = {v: k for k, v in case_id_to_pid_map.items()}

        for pid_nii in self.patient_ids:
            pid = pid_nii.split('.nii')[0]
            if pid in pid_to_case_id:
                case_id = int(pid_to_case_id[pid])
                # 寻找对应的文件名，例如 TumorTotal_005_0000.nii.gz
                fname = next(images_dir.glob(f"*_{case_id:03d}_*.nii.gz"), None)
                if fname and fname.exists():
                    self.fnames.append(fname)
                    self.label_array.append(ALL_PATIENTS_LABELS[pid_nii])
                else:
                    print(f"警告: 找不到PID {pid} 对应的图像文件。")
            else:
                print(f"警告: 在pid_map中找不到PID {pid}。")

        # 5倍扩增 (仅在训练模式下，对训练集生效)
        if self.mode == 'train':
            self.fnames = self.fnames * 5
            self.label_array = self.label_array * 5
        
        self.label_array = np.array(self.label_array, dtype=np.int64)
        print(f"模式: '{self.mode}', 基于 {len(self.patient_ids)} 个病人ID，成功加载 {len(self.fnames)} 个样本。")
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        fname = self.fnames[index]
        label = self.label_array[index]

        # 1. 加载数据为 Numpy 数组
        numpy_image = itk.GetArrayFromImage(itk.imread(str(fname), itk.F)) # (Z, H, W)
        frame_count = numpy_image.shape[0]

        # 2. 时间维度的裁剪 (Temporal Crop)
        #    我们仍然需要这一步来确保输入到3D模型的序列长度是固定的
        frame_indices = list(range(frame_count))
        
        # 循环填充 (Loop Padding) 以处理短于 temporal_size 的序列
        if frame_count < self.temporal_size:
            diff = self.temporal_size - frame_count
            for i in range(diff):
                frame_indices.append(frame_indices[i % frame_count])

        # 时间裁剪
        if self.mode == 'train': # 随机裁剪
            start_idx = np.random.randint(0, len(frame_indices) - self.temporal_size + 1)
        else: # 中心裁剪
            start_idx = (len(frame_indices) - self.temporal_size) // 2
        
        frame_indices = frame_indices[start_idx : start_idx + self.temporal_size]
        
        # 根据裁剪后的索引选取帧
        volume = numpy_image[frame_indices] # (T, H, W) T=temporal_size

        # 3. 将单通道灰度数据转换为三通道 RGB
        #    Albumentations 需要 (H, W, C) 格式
        #    我们将 (T, H, W) 转换为 (T, H, W, 3)
        volume_rgb = np.repeat(volume[..., np.newaxis], 3, axis=-1).astype(np.uint8)

        # 4. [核心] 应用 Albumentations 变换
        if self.albumentations_transform:
            # 为了让所有帧应用相同的几何变换，我们使用 additional_targets
            # 将第一帧作为'image'，其余帧作为'image1', 'image2'...
            images_dict = {'image': volume_rgb[0]}
            for i in range(1, self.temporal_size):
                images_dict[f'image{i}'] = volume_rgb[i]
            
            # 定义变换要作用于哪些目标
            self.albumentations_transform.add_targets({f'image{i}': 'image' for i in range(1, self.temporal_size)})
            
            # 执行变换
            transformed = self.albumentations_transform(**images_dict)
            
            # 将变换后的结果重新堆叠起来
            transformed_slices = [transformed['image']]
            for i in range(1, self.temporal_size):
                transformed_slices.append(transformed[f'image{i}'])
            
            # 堆叠后，ToTensorV2 会自动将 (T, H, W, C) -> (C, T, H, W)
            tensor_volume = torch.stack(transformed_slices, dim=1)
        else:
            # 如果没有变换，手动转换为Tensor
            tensor_volume = torch.from_numpy(volume_rgb).permute(3, 0, 1, 2).float() / 255.0

        return tensor_volume, label