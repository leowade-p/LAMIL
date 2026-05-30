

from pathlib import Path 
import pylidc as pl
import numpy as np 
import torchio as tio 
import pandas as pd 
from tqdm import tqdm
from multiprocessing import Pool, Manager
from pylidc.utils import consensus

LABELS = ['subtlety', 'internalStructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation',
          'texture', 'malignancy']


def scan2labels(scan_id):
    scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()
    
    # Read nifti (required for correct affine matrix)
    path_rel = Path(scan.get_path_to_dicom_files()).relative_to(path_root_download)
    vol = tio.ScalarImage(path_root_data/path_rel/'img.nii.gz') 

    scan_ann = []

    for nod_idx, nodules in enumerate(scan.cluster_annotations()): # Each scan has multiple nodules
        for ann_idx, ann in enumerate(nodules): # Each nodule was rated between 1 and 4 raters
            ann_dict = {label:getattr(ann, label) for label in LABELS}
            ann_dict['bbox'] = [[d.start, d.stop] for d in  ann.bbox()]
            ann_dict['scan_id'] = scan.id # equal for all nodules/annotations 
            ann_dict['nodule_idx'] = nod_idx 
            ann_dict['annotation_idx'] = ann_idx 
            ann_dict['annotation_num'] = len(nodules) 
            ann_dict['annotation_id'] = ann.id # unique - same annotator has different numbers 
            ann_dict['patient_id'] = scan.patient_id
            ann_dict['study_instance_uid'] = scan.study_instance_uid
            ann_dict['series_instance_uid'] = scan.series_instance_uid
            scan_ann.append(ann_dict)
            
             
            mask_vol = np.zeros(vol.spatial_shape, dtype=np.uint8)
            bbox = ann.bbox()
            mask = ann.boolean_mask()
            
            mask_vol[bbox][mask] = 1 
            mask_vol = tio.LabelMap(tensor=mask_vol[None], affine=vol.affine)
            mask_vol.save(path_root_data/path_rel/f'seg_{nod_idx}_{ann_idx}.nii.gz')

        # Perform a consensus consolidation and 50% agreement level.
        cmask,cbbox, masks = consensus(nodules, clevel=0.5)
        mask_vol = np.zeros(vol.spatial_shape, dtype=np.uint8)
        mask_vol[cbbox][cmask] = 1 
        mask_vol = tio.LabelMap(tensor=mask_vol[None], affine=vol.affine)
        mask_vol.save(path_root_data/path_rel/f'seg_{nod_idx}.nii.gz')

        
    return scan_ann


if __name__ == "__main__":
    scan_ids = range(1, len(list(pl.query(pl.Scan)))+1)
    
    # Settings 
    path_root = Path('/home/gustav/Coscine_Public/LIDC-IDRI')
    path_root_download = path_root/'download/TCIA_LIDC-IDRI_20200921/LIDC-IDRI'
    path_root_out = path_root/'preprocessed'
    path_root_data = path_root_out/'data'


    # Option 1: Multi-CPU 
    all_ann = []
    with Pool() as pool:
        for scan_ann in tqdm(pool.imap_unordered(scan2labels, scan_ids), total=len(scan_ids)):
            all_ann.extend(scan_ann)

    # Option 2: Single-CPU (if you need a coffee break)
    # all_ann = []
    # for scan_id in tqdm(scan_ids):
    #     all_ann.extend(scan2labels(scan_id))

    df = pd.DataFrame(all_ann)
    df.to_csv(path_root_out/'annotation.csv', index=False)