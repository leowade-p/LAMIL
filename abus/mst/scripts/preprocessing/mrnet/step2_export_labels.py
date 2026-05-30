from pathlib import Path 
import torchio as tio 
import numpy as np 
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd 

path_root = Path('/home/gustav/Coscine_Public/MRNet/')
path_root_in = path_root/'download/MRNet-v1.0'
path_root_out = path_root/'preprocessed/'
path_root_out.mkdir(parents=True, exist_ok=True) 



if __name__ == '__main__':
    df_train = pd.DataFrame()
    for pathology in ['abnormal', 'acl', 'meniscus']:
        df = pd.read_csv(path_root_in/f'train-{pathology}.csv', names=['ID', pathology])
        df_train = pd.merge(df_train, df, on='ID') if len(df_train)>0 else df

    df_train.to_csv(path_root_out/'train.csv', index=False)

    df_val = pd.DataFrame()
    for pathology in ['abnormal', 'acl', 'meniscus']:
        df = pd.read_csv(path_root_in/f'valid-{pathology}.csv', names=['ID', pathology])
        df_val = pd.merge(df_val, df, on='ID') if len(df_val)>0 else df

    df_val.to_csv(path_root_out/'valid.csv', index=False)