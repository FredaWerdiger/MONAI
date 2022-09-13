# measuring lesion volumes for sub groups

import os
import SimpleITK as sitk
import pandas as pd
import glob
import numpy as np
from sklearn.cluster import k_means
from scipy.stats import ttest_ind

path_to_folders = 'D:/ctp_project_data/DWI_Training_Data/'

test = path_to_folders + 'test/'
groups = ['train', 'val', 'test']

results = pd.read_csv('D:/ctp_project_data/DWI_Training_Data/out_final/results.csv')


def get_volumes(group):
    masks = glob.glob(path_to_folders + group + '/masks/*')
    ims = [sitk.ReadImage(mask) for mask in masks]
    volumes =[(np.prod([float(im.GetMetaData('pixdim[' + dim + ']'))
                             for dim in ['1','2','3']], axis=0)/1000) *
                     sitk.GetArrayFromImage(im).sum()
                    for im in ims]
    ids = [group + '_' + str(i+1).zfill(3) for i, _ in enumerate(volumes)]
    labels = k_means(np.reshape(volumes, (-1, 1)), n_clusters=2,
            random_state=0)[1]
    df = pd.DataFrame(columns=['volume_ml', 'volume_group'], index=ids)
    df['volume_ml'] = volumes
    df['volume_group'] = labels
    return df

test_df = get_volumes('test')
train_df = get_volumes('train')
val_df = get_volumes('validation')
test_df_join = results[['id', 'dice']].join(test_df, how='right', on='id')


small = test_df_join[test_df_join.apply(lambda row: row.volume_ml < 70, axis=1)]
dice_small = small.dice

big = test_df_join[test_df_join.apply(lambda row: row.volume_ml > 70, axis=1)]
dice_big = big.dice

ttest_ind(dice_small.values, dice_big.values) # not convincing

