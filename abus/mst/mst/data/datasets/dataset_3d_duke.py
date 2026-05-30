from pathlib import Path 
import pandas as pd 
import torch.utils.data as data 
import torchio as tio
import torch

from .augmentations.augmentations_3d import ImageOrSubjectToTensor, RescaleIntensity, ZNormalization, CropOrPad


class DUKE_Dataset3D(data.Dataset):
    PATH_ROOT = Path('/home/gustav/Documents/datasets/Duke-Breast-Cancer-MRI/')
    LABEL = 'Malignant'

    def __init__(
            self,
            path_root=None,
            fold = 0,
            split= None,
            fraction=None,
            transform = None,
            image_resize = None,
            resample=None,
            flip = False,
            random_rotate=False,
            image_crop = (224, 224, 32),
            random_center=False,
            noise=False, 
            to_tensor = True,
        ):
        self.path_root = self.PATH_ROOT if path_root is None else Path(path_root)
        self.path_root_data = self.path_root/'preprocessed_crop/data'
        self.split = split 

        if transform is None: 
            self.transform = tio.Compose([
                tio.Resize(image_resize) if image_resize is not None else tio.Lambda(lambda x: x),
                tio.Resample(resample) if resample is not None else tio.Lambda(lambda x: x),
                tio.Flip(1), # Just for viewing, otherwise upside down
                CropOrPad(image_crop, random_center=random_center, padding_mode='minimum') if image_crop is not None else tio.Lambda(lambda x: x),
                ZNormalization(per_channel=True, per_slice=False, masking_method=lambda x:(x>x.min()) & (x<x.max()), percentiles=(0.5, 99.5)),   # 0.5, 99.5   2.5, 97.5
                # tio.Lambda(lambda x: x.moveaxis(1, 2) if torch.rand((1,),)[0]<0.5 else x ) if random_rotate else tio.Lambda(lambda x: x), # WARNING: 1,2 if Subject, 2, 3 if tensor
                tio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, 0,90), translation=0, isotropic=True, default_pad_value='minimum') if random_rotate else tio.Lambda(lambda x: x),
                tio.RandomFlip((0,1,2)) if flip else tio.Lambda(lambda x: x), # WARNING: Padding mask 
                tio.Lambda(lambda x:-x if torch.rand((1,),)[0]<0.5 else x, types_to_apply=[tio.INTENSITY]) if noise else tio.Lambda(lambda x: x),
                tio.RandomNoise(std=(0.0, 0.25)) if noise else tio.Lambda(lambda x: x),

                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x)             
            ])
        else:
            self.transform = transform


        # Get split file 
        path_csv = self.path_root/'preprocessed_crop/splits/split.csv'
        path_or_stream = path_csv 
        self.df = self.load_split(path_or_stream, fold=fold, split=split, fraction=fraction)
        self.item_pointers = self.df.index.tolist()


    def __len__(self):
        return len(self.item_pointers)

    def load_img(self, path_img):
        return tio.ScalarImage(path_img)

    def load_map(self, path_img):
        return tio.LabelMap(path_img)

    def __getitem__(self, index):
        idx = self.item_pointers[index]
        item = self.df.loc[idx]
        target = item[self.LABEL]
        uid = item['UID']

        img = self.load_img(self.path_root_data/f'Breast_MRI_{uid}'/'sub.nii.gz')
        img = self.transform(img)

        return {'uid':uid, 'source': img, 'target':target}


    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        df = pd.read_csv(filepath_or_buffer)
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]   
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index()
        return df