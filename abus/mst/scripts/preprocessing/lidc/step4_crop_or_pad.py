from pathlib import Path 
import torchio as tio 
import torch
import numpy as np 
from multiprocessing import Pool
from tqdm import tqdm
from itertools import chain

def preprocess(path_file):
   
    path_dir = path_file.parent 
    img_ref = tio.ScalarImage(path_file)

    # Create output directory 
    path_out_dir = path_out/path_dir.relative_to(path_in)
    path_out_dir.mkdir(exist_ok=True, parents=True)

   
    for path_seg in chain(path_dir.glob('seg_[0-9].nii.gz'), path_dir.glob('seg_[0-9][0-9].nii.gz')) : # There are max. 23 nodules per series
        nod_idx = int(path_seg.name.split('.')[0].split('_')[1])
        base_name = path_seg.name.split('.')[0]
        transform = tio.CropOrPad((256, 256, 32), mask_name='mask', padding_mode=-1024)
        mask = tio.LabelMap(path_seg)
        mask_raters = {path_seg_rater.name:tio.LabelMap(path_seg_rater) for path_seg_rater in path_dir.glob(f'{base_name}_*.nii.gz')}
        subject = tio.Subject(img=img_ref, mask=mask, **mask_raters)
        
        subject = transform(subject)

        subject['img'].save(path_out_dir/f'img_{nod_idx}.nii.gz')
        subject['mask'].save(path_out_dir/f'seg_{nod_idx}.nii.gz')
        for filename in mask_raters.keys():
            subject[filename].save(path_out_dir/filename)


if __name__ == "__main__":
    path_root = Path('/home/gustav/Coscine_Public/LIDC-IDRI/')
    path_in = path_root/'preprocessed/data'
    path_out = path_root/'preprocessed_crop/data'
    path_out.mkdir(parents=True, exist_ok=True)
    files = list(path_in.rglob('img.nii.gz'))  # Convert the iterator to a list

    # Option 1: Multi-CPU
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(preprocess, files), total=len(files)):
            pass

    # Option 2: Single-CPU (if you need a coffee break)
    # for file in tqdm(files):
    #     preprocess(file)