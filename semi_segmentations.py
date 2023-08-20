import os
import pandas as pd
import glob
import SimpleITK as sitk
from sklearn.metrics import f1_score


HOMEDIR = os.path.expanduser('~/')
mediaflux = HOMEDIR + 'mediaflux/'

if not os.path.exists(mediaflux):
    mediaflux = 'Z:/'

directory = mediaflux + 'data_freda/ctp_project/CTP_DL_Data/'
semi_masks = [file for file in glob.glob(directory + 'no_seg/masks_semi/*')
              if ('exclude' not in file and 'funny' not in file)]
masks = [file for file in glob.glob(directory + 'no_seg/masks/*')
         if 'exclude' not in file]
subjects = ['INSP_' + sub.split('.nii')[0].split('_')[-1] for sub in semi_masks]

# # running for cases that james and chrissy did
# masks = [file for file in glob.glob(mediaflux  + 'data_freda/dl_seg_check/*pred*')]
# semi_masks = [file for file in glob.glob(mediaflux  + 'data_freda/dl_seg_check/*seg*')]

# subjects = ['INSP_' + sub.split('.nii')[0].split('_')[-2] for sub in semi_masks]
subjects.sort()

df = pd.DataFrame(columns=["subject",
                           "volume_gt",
                           "volume_pred",
                           "abs_diff",
                           "perc_diff",
                           "dsc",
                           "overcall",
                           "undercall",
                           "blood",
                           ])
df.subject = subjects

for sub in subjects:
    print(f"Running for {sub}")
    semi_mask = sitk.ReadImage([file for file in semi_masks if sub in file][0])
    spacing = semi_mask.GetSpacing()
    voxel_ml = spacing[0] * spacing[1] * spacing[2] / 1000
    labelStats = sitk.LabelShapeStatisticsImageFilter()

    labelStats.SetBackgroundValue(0)
    labelStats.Execute(semi_mask)
    try:
        size_lesion = len(labelStats.GetIndexes(1)) * voxel_ml
    except RuntimeError:
        size_lesion = 0.
    df.loc[df.subject == sub, 'volume_gt'] = round(size_lesion, 1)

    mask = sitk.Cast(
        sitk.ReadImage([file for file in masks if sub in file][0]), sitk.sitkUInt8)
    labelStats.Execute(mask)
    try:
        size_prediction = len(labelStats.GetIndexes(1)) * voxel_ml
    except RuntimeError:
        size_prediction = 0.
    df.loc[df.subject == sub, 'volume_pred'] = round(size_prediction, 1)

    diff = size_prediction - size_lesion
    if diff < 0:
        df.loc[df.subject == sub, 'undercall'] = 1
        df.loc[df.subject == sub, 'overcall'] = 0
    elif diff > 0:
        df.loc[df.subject == sub, 'undercall'] = 0
        df.loc[df.subject == sub, 'overcall'] = 1
    else:
        df.loc[df.subject == sub, 'undercall'] = 0
        df.loc[df.subject == sub, 'overcall'] = 0

    if not size_lesion == 0:
        perc = abs(diff)/size_lesion * 100
    else:
        perc = 0

    df.loc[df.subject == sub, 'abs_diff'] = round(abs(diff), 1)
    df.loc[df.subject == sub, 'perc_diff'] = round(perc, 1)

    # dice
    array_semi = sitk.GetArrayFromImage(semi_mask).flatten()
    array_mask = sitk.GetArrayFromImage(mask).flatten()
    f1 = f1_score(array_mask, array_semi)
    df.loc[df.subject == sub, 'dsc'] = round(f1, 3)

df.sort_values(by='subject', inplace=True)

# df.to_csv('../study_design/study_lists/no_seg_stats.csv', index=False)
df.to_csv('../study_design/study_lists/no_seg_stats_densenet.csv', index=False)

keys = ["volume_gt", "volume_pred", "abs_diff", "perc_diff", "dsc"]

for key in keys:
    df[key] = df[key].apply(pd.to_numeric, errors='coerce')

df_blood = pd.read_csv('../study_design/study_lists/no_seg_stats_with_blood.csv')


import seaborn as sns

import matplotlib.pyplot as plt

fig = plt.figure()
ax = sns.scatterplot(
    x=df_blood.subject.to_list(),
    y=df_blood.abs_diff.to_list(),
    hue=df_blood.blood.to_list())
plt.legend(["any blood", "no blood"])

ax.set(ylabel="volume diff", ylim=(0, 100), xticks='')
plt.show()

