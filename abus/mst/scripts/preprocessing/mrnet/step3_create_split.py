from pathlib import Path 
import numpy as np 
import pandas as pd 

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

path_root = Path('/home/gustav/coscine_public/MRNet/preprocessed')
path_root_out = path_root/'splits'
path_root_out.mkdir(parents=True, exist_ok=True)

df_train = pd.read_csv(path_root/'train.csv')
df_valid = pd.read_csv(path_root/'valid.csv')

print("Number train.csv: ", len(df_train), " of 1130")
print("Number valid.csv: ", len(df_valid), " of 120")

for cls in ['abnormal', 'acl', 'meniscus']:
    print("Tumor train: ", df_train[cls].value_counts(normalize=True))
    print("Tumor valid: ", df_valid[cls].value_counts(normalize=True))
    print()



df_train = df_train.reset_index(drop=True)
splits = []
sgkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) 
for fold_i, (train_idx, val_idx) in enumerate(sgkf.split(df_train['ID'], df_train['abnormal'])):
    df_split = df_train.copy()
    df_split['Fold'] = fold_i 
    df_split['Folder'] = 'train/'
    train_idx, val_idx = df_train.iloc[train_idx].index, df_train.iloc[val_idx].index 
    df_split.loc[train_idx, 'Split'] = 'train' 
    df_split.loc[val_idx, 'Split'] = 'val' 

    df_valid_copy = df_valid.copy()
    df_valid_copy['Fold'] = fold_i
    df_valid_copy['Folder'] = 'valid/'
    df_valid_copy['Split'] = 'test'
    df_split = pd.concat([df_split, df_valid_copy])
    splits.append(df_split)

    # Stop here, test set remains anyway the same 
    break 
df_splits = pd.concat(splits)


df_splits.to_csv(path_root_out/'split.csv', index=False)