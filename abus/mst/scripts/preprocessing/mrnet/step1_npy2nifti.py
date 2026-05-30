from pathlib import Path 
import torchio as tio 
import numpy as np 
from tqdm import tqdm
from multiprocessing import Pool

path_root = Path('/home/gustav/Coscine_Public/MRNet/')
path_root_in = path_root/'download/MRNet-v1.0'
path_root_out = path_root/'preprocessed/data'
path_root_out.mkdir(parents=True, exist_ok=True) 



def npy2nifti(path_file):
    # Read
    data = np.load(path_file)

    # To Nifti 
    img = tio.ScalarImage(tensor=data[None])

    # Write
    filestem = path_file.stem 
    path_out_dir = path_root_out/path_file.parent.relative_to(path_root_in)
    path_out_dir.mkdir(parents=True, exist_ok=True)
    img.save(path_out_dir/f'{filestem}.nii.gz')


if __name__ == '__main__':

    path_files = list(path_root_in.rglob('*npy'))

    # Option 1: Multi-CPU 
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(npy2nifti, path_files), total=len(path_files)):
            pass

    # Option 2: Single-CPU (if you need a coffee break)
    # for path_file in tqdm(path_files):
    #     npy2nifti(path_file)
