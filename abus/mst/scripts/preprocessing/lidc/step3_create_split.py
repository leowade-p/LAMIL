from pathlib import Path 
import numpy as np 
import pandas as pd 

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

path_root = Path('/home/gustav/Coscine_Public/LIDC-IDRI/preprocessed')
df = pd.read_csv(path_root/'annotation.csv')

# Note:  875 patients have a Nodule>=3mm 
# df_nodules = pd.read_excel(path_root.parent/'download/lidc-idri-nodule-counts-6-23-2015.xlsx').iloc[:-1]
# print(len(df_nodules[df_nodules['Number of Nodules >=3mm**']>0]['TCIA Patent ID'].unique()))
# print(df_nodules[df_nodules['Number of Nodules >=3mm**']>0]['Number of Nodules >=3mm**'].sum())
print("Number Annotations: ", len(df))
print("Number Nodules (>3mm)", len(np.unique(df[['scan_id', 'nodule_idx']])) )
print("Number Series: ", len(np.unique(df[ 'series_instance_uid'])))
print("Number Patients: ", len(np.unique(df[ 'patient_id'])))

unique_cols = ['patient_id', 'study_instance_uid', 'series_instance_uid', 'scan_id', 'nodule_idx']
df1 = df.groupby(unique_cols)['malignancy'].apply(lambda x:int(x.mean().round())).reset_index()

df2 = df.drop_duplicates(unique_cols)
df2 = df2.drop(columns='malignancy')

df = pd.merge(df1, df2, on=unique_cols).reset_index(drop=True)

df = df[df['malignancy']!=3] # Remove all uncertain cases # See different approaches eg. https://doi.org/10.1038%2Fs41598-018-27569-w
df['Malignant'] = (df['malignancy'] > 3).astype(int)

print("Malignant: ", df['Malignant'].value_counts())


#1: ‘Highly Unlikely’
#2: ‘Moderately Unlikely’
#3: ‘Indeterminate’
#4: ‘Moderately Suspicious’
#5: ‘Highly Suspicious’

df = df.reset_index(drop=True)
splits = []
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0) # StratifiedGroupKFold
sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
for fold_i, (train_val_idx, test_idx) in enumerate(sgkf.split(df['scan_id'], df['Malignant'], groups=df['patient_id'])):
    df_split = df.copy()
    df_split['Fold'] = fold_i 
    df_trainval = df_split.loc[train_val_idx]
    train_idx, val_idx = list(sgkf2.split(df_trainval['scan_id'], df_trainval['Malignant'], groups=df_trainval['patient_id']))[0]
    train_idx, val_idx = df_trainval.iloc[train_idx].index, df_trainval.iloc[val_idx].index 
    df_split.loc[train_idx, 'Split'] = 'train' 
    df_split.loc[val_idx, 'Split'] = 'val' 
    df_split.loc[test_idx, 'Split'] = 'test' 
    splits.append(df_split)
df_splits = pd.concat(splits)

path_root_out = path_root/'splits'
path_root_out.mkdir(parents=True, exist_ok=True)
df_splits.to_csv(path_root_out/'split.csv', index=False)