import os
import cv2
import numpy as np
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 完整标签字典
LABEL_DICT = {'2195561.nii': 1, '1918325.nii': 1, '1916513.nii': 1, '1852423.nii': 1, '1917881.nii': 1,
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


class TumorDataset(Dataset):
    def __init__(self, root, transform: Optional[A.Compose] = None, file_list: Optional[list[str]] = None):
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, "train2017")
        self.mask_dir = os.path.join(root, "panoptic_train2017")
        self.image_files = file_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask == 255).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask


def get_dataloader(
    file_list, 
    img_dim=(392, 392),
    batch_size=6,
    root="/home/huyiding/pengdie/dino/det_datasets/4v1",
    num_workers=16,
    pin_memory=True,
    persistent_workers=False,
    is_train = True,
):
    file_list = sorted(file_list)
    if(is_train == True):
        transform = A.Compose([
        # 保持比例缩放，并 pad 成 392x392
        A.LongestMaxSize(max_size=392),
        A.PadIfNeeded(min_height=392, min_width=392, border_mode=0, value=0),

        # 几何增强（同时作用于 image 和 mask）
        A.Rotate(limit=10, p=0.2),
        A.RandomResizedCrop(size=(392,392), scale=(0.9, 1.0), ratio=(0.9, 1.1), p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # 图像增强（只作用于 image）
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
        # 转 tensor（图像 float32、掩膜 long/int）
        ToTensorV2(),
        ])
    else :
        transform = A.Compose([
        # 保持比例缩放，并 pad 成 392x392
        A.LongestMaxSize(max_size=392),
        A.PadIfNeeded(min_height=392, min_width=392, border_mode=0, value=0),
        # 图像增强（只作用于 image）
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
        # 转 tensor（图像 float32、掩膜 long/int）
        ToTensorV2()
        ])



    dataset = TumorDataset(root, transform=transform, file_list=file_list)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # 外面调用时可以控制 train/val 是否 shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    print(f"✅ Loaded {len(file_list)} files from {root}/train2017")

    return loader
