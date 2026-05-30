

from pathlib import Path 
import logging 
import sys
import pandas as pd 
import pydicom.datadict
import pydicom.dataelem
import pydicom.sequence
import pydicom.valuerep
from tqdm import tqdm
import torchio as tio 
import torch 
import pydicom
import pylidc as pl
from multiprocessing import Pool





def maybe_convert(x):
    if isinstance(x, pydicom.sequence.Sequence):
        # return [maybe_convert(item) for item in x]
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.dataset.Dataset):  
        # return dataset2dict(x)
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    else:
        return x 


def dataset2dict(ds, exclude=['PixelData', '']):
    return {keyword:value for key in ds.keys() 
            if ((keyword := ds[key].keyword) not in exclude)  and ((value := maybe_convert(ds[key].value)) is not None) }


def scan2nifti(scan_id):
    scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()

    # Get path to series
    path_series = Path(scan.get_path_to_dicom_files())

    # Read DICOM
    img_pl = scan.to_volume()
    affine = torch.zeros((4,4))
    affine[0, 0] = scan.spacings[0]
    affine[1, 1] = scan.spacings[1]
    affine[2, 2] = scan.spacings[2]
    img_tio = tio.ScalarImage(tensor=img_pl[None], affine=affine)

    # Read Metadata 
    ds = pydicom.dcmread(next(path_series.glob('*.dcm'), None), stop_before_pixels=True)
    metadata = dataset2dict(ds)
    
    # Create output folder 
    rel_path = path_series.relative_to(path_root_in)
    path_out_dir = path_root_out_data/rel_path
    path_out_dir.mkdir(exist_ok=True, parents=True)

    # Write 
    filename = 'img.nii.gz' #metadata['ProtocolName']
    logger.info(f"Writing file: {filename}:")
    img_tio.save(path_out_dir/filename )

    # Add additional information 
    metadata['_SpatialShape'] = list(img_tio.spatial_shape)
    metadata['_Path'] = str(rel_path/filename)

    return metadata





if __name__ == "__main__":
    # WARNING: DON'T try to read DICOM yourself - LIDC is messy  :( 
    # Use pylidc, as it "fixes", for example "Some scans contain multiple slices with the same `z` coordinate" - wtf  
    # Follow instructions: https://pylidc.github.io/install.html

    # Setting 
    path_root = Path('/home/gustav/Coscine_Public/LIDC-IDRI')
    path_root_in = path_root/'download/TCIA_LIDC-IDRI_20200921/LIDC-IDRI'
    path_root_out = path_root/'preprocessed'
    path_root_out_data = path_root_out/'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)


    # Logging 
    path_log_file = path_root_out/'preprocessing.log'
    logger = logging.getLogger(__name__)
    s_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(path_log_file, 'w')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[s_handler, f_handler])
    

    # Get all scans 
    scan_ids = range(1, len(list(pl.query(pl.Scan)))+1)

    # Option 1: Multi-CPU 
    metadata_list = []
    with Pool() as pool:
        for meta in tqdm(pool.imap_unordered(scan2nifti, scan_ids), total=len(scan_ids)):
            metadata_list.append(meta)

    # Option 2: Single-CPU (if you need a coffee break)
    # metadata_list = []
    # for scan_id in tqdm(scan_ids):
    #     meta = scan2nifti(scan_id)
    #     metadata_list.append(meta)

    # Check export 
    path_exports = [path.relative_to(path_root_out) for path in path_root_out.rglob('img.nii.gz')]
    num_patients = list(set([path.parts[0] for path in  path_exports]))
    print("Exported Patients:", len(num_patients), " of 1010")
    print("Exported Studies:", len(path_exports), " of 1018 (pylidc) or 1308 (TCIA)")

    # Save metadata 
    df = pd.DataFrame(metadata_list)
    df.to_csv(path_root_out/'metadata.csv', index=False)