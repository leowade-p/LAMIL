# 良性数据处理2v1
# import SimpleITK as sitk
# import cv2
import matplotlib
# matplotlib.use('TkAgg')
import nibabel as nib
import numpy as np
import os
from shutil import copy, rmtree
import pydicom
from PIL import Image
import pandas as pd

# 验证集 36例
test_benign = ['1926144.nii', '2027345.nii', '1850473.nii', '2336756.nii', '2701356.nii', 
               '2125006.nii', '1925991.nii', '2582916.nii', '1132044.nii', '2639837.nii', 
               '2770800.nii', '2698551.nii', '1915205.nii', '2110802.nii', '2027347.nii', 
               '1270467.nii', '2042416.nii', '3237293.nii', '1925620.nii', '2100216.nii', 
               '2577307.nii', '1915207.nii', '2736060.nii', '1927969.nii', '1915238.nii', 
               '1953114.nii', '2760906.nii', '2093628.nii', '1851152.nii', '2699066.nii', 
               '1882547.nii', '1921022.nii', '1864666.nii', '2770801.nii', '2701358.nii', '1925759.nii']

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # rmtree(file_path)
        # os.makedirs(file_path)
        return file_path
    os.makedirs(file_path)
    return file_path

img_path = "../../良性肿瘤增强部分/ini"
mask_path = "../../良性肿瘤增强部分/roi"

# 原始图片的病人文件夹名
img_class = [cla for cla in os.listdir(img_path) if cla.endswith("nii")]
# mask的文件名
mask_class = [cla for cla in os.listdir(mask_path) if cla.endswith("nii")]

root = "./det_datasets/2v1"
train_img = mk_file(os.path.join(root, "train2017"))
val_img = mk_file(os.path.join(root, "val2017"))
train_mask = mk_file(os.path.join(root, "panoptic_train2017"))
val_mask = mk_file(os.path.join(root, "panoptic_val2017"))

def normalizatingImage(img, fc, fw):
    fmin = (2.0 * fc - fw) / 2 + 0.5
    fmax = (2.0 * fc + fw) / 2 + 0.5
    img = np.clip(img, fmin, fmax)
    img = (img - fmin) / (fmax - fmin) * 255
    img = Image.fromarray(img)
    if img.mode == "F":
        img = img.convert('RGB')
    return img

# # 遍历原始数据并保存有效的slice切片
for index, cla in enumerate(img_class):
    mask_name = cla

    if mask_name in mask_class:
        # 读取原图dicom文件 以及mask标签nii文件，都为原始的数组形式
        img_in = os.path.join(img_path, cla)
        mask_in = os.path.join(mask_path, mask_name)

        img_original = nib.load(img_in).get_fdata()  # 三维数组  x,y,z
        img_original = np.transpose(img_original)  # z, y, x,

        fc = -100
        fw = 400

        mask = nib.load(mask_in).get_fdata()  # mask的三维数组  x,y,z
        mask = np.transpose(mask)  # z, y, x,

        # 判断mask和img的尺寸是否匹配
        if mask.shape == img_original.shape:
            # 找出有mask的标签slice图片范围 以z 方向进行切片
            for i in range(mask.shape[0]):
                
                if (mask[i, :, :] == 0).all():
                    continue
                else:
                    mask_slice = mask[i, :, :] * 255
                    mask_slice = Image.fromarray(mask_slice).convert('L')

                    img_slice = img_original[i, :, :]
                    fc = -100
                    fw = 400
                    img_slice0 = normalizatingImage(img_slice, fc, fw).convert('L')

                    img_slice2 = normalizatingImage(img_slice, 28, 277).convert('L')
                    img_slice3 = normalizatingImage(img_slice, 109, 527).convert('L')
                    img_slice4 = normalizatingImage(img_slice, -86, 598).convert('L')

                    result_image = Image.merge('RGB', [img_slice2, img_slice3, img_slice4])

                    img_slice0 = np.array(img_slice0)
                    result_image = np.array(result_image)
                    mask_slice = np.array(mask_slice)

                    if (img_slice0 <= 10).all():
                        continue
                    else:
                        pos = np.where(img_slice0 > 10)
                        result_image = result_image[np.min(pos[0]):np.max(pos[0]), np.min(pos[1]):np.max(pos[1]), :]
                
                        # img_slice0 = img_slice0[np.min(pos[0]):np.max(pos[0]), np.min(pos[1]):np.max(pos[1])]
                
                        mask_slice = mask_slice[np.min(pos[0]):np.max(pos[0]), np.min(pos[1]):np.max(pos[1])]

                        hw = 400
                        h, w, c = result_image.shape
                        if h < hw:
                            result_image = np.pad(result_image, (((hw-h)//2,hw-h-(hw-h)//2), (0,0), (0,0)))
                            mask_slice = np.pad(mask_slice, (((hw-h)//2,hw-h-(hw-h)//2), (0,0)))
                            # img1 = np.pad(img, (((hw-h)//2,hw-h-(hw-h)//2), ((hw-w)//2,hw-w-(hw-w)//2), (0,0)))
                        if w < hw:
                            result_image = np.pad(result_image, ((0,0), ((hw-w)//2,hw-w-(hw-w)//2), (0,0)))
                            mask_slice = np.pad(mask_slice, ((0,0), ((hw-w)//2,hw-w-(hw-w)//2)))
    
                        # img_slice0 = Image.fromarray(img_slice0)
                        result_image = Image.fromarray(result_image)
                        mask_slice = Image.fromarray(mask_slice)

                        if cla in test_benign:
                            result_image.save(f"{os.path.join(val_img, f'{img_class[index][:-4]}_{i}.png')}")
                            mask_slice.save(f"{os.path.join(val_mask, f'{img_class[index][:-4]}_{i}.png')}")
                        else:
                            result_image.save(f"{os.path.join(train_img, f'{img_class[index][:-4]}_{i}.png')}")
                            mask_slice.save(f"{os.path.join(train_mask, f'{img_class[index][:-4]}_{i}.png')}")

        else:
            print(f'病人集{cla}的原始 img：{img_original.shape} 与 mask: {mask.shape}维度不匹配！')

    else:
        print(f'{cla} 没有相对应的mask文件！')

print(index)
print("done!")