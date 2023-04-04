# this code is to compare MiStar map to my own prediction
import os

import numpy as np
import pandas as pd
import glob
import SimpleITK as sitk
from sklearn.metrics import f1_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

def main(out_tag):
    HOMEDIR = os.path.expanduser('~/')
    if os.path.exists(HOMEDIR + 'mediaflux/'):
        mediaflux = HOMEDIR + 'mediaflux/'
        directory = HOMEDIR + 'mediaflux/data_freda/ctp_project/CTP_DL_Data/'
        ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])
        atlas_df = pd.read_excel(HOMEDIR + 'PycharmProjects/study_design/ATLAS_clinical_20221006_1304.xlsx',
                                 sheet_name='Sheet1',
                                 header=[0],
                                 usecols=['INSPIRE ID', 'Occlusion severity (TIMI:0=complete occlusion, 3=normal)'])
    elif os.path.exists('Z:/data_freda'):
        mediaflux = 'Z:'
        directory = 'Z:/data_freda/ctp_project/CTP_DL_Data/'
        ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])
        atlas_df = pd.read_excel(HOMEDIR + 'PycharmProjects/study_design/ATLAS_clinical_20221006_1304.xlsx',
                                 sheet_name='Sheet1',
                                 header=[0],
                                 usecols=['INSPIRE ID', 'Occlusion severity (TIMI:0=complete occlusion, 3=normal)'])

    print(f"out_tag = {out_tag}")

    results_folder = os.path.join(directory, 'out_' + out_tag)
    results_csv = glob.glob(results_folder + '/results*.csv')[0]
    results_df = pd.read_csv(results_csv)
    results_df['mistar_core'] = ''
    results_df['mistar_penumbra'] = ''
    results_df['mistar_dice'] = ''

    gt_folder = os.path.join(directory, 'DATA', 'masks')
    pred_folder = os.path.join(results_folder, 'pred')

    for subject in results_df.subject.to_list():
        dl_id = str(results_df.loc[results_df.subject == subject, 'id'].values[0]).zfill(3)

        mistar_dir = mediaflux + '/INSPIRE_database/' + subject + '/CT_baseline/CTP_baseline/mistar/'
        mistar_lesion = glob.glob(mistar_dir + '*' + 'Lesion.nii.gz')[0]
        mistar_img = sitk.ReadImage(mistar_lesion)
        mistar_array = sitk.GetArrayFromImage(mistar_img).astype(float)
        # get core and penumbra
        core = (mistar_array == 220) * 1
        core = np.asarray(core)
        # get gt
        gt = glob.glob(gt_folder + '/*' + dl_id + '*')[0]
        gt_array = sitk.GetArrayFromImage(sitk.ReadImage(gt))
        dice_mistar = f1_score(gt_array.flatten(), core.flatten())
        # get volumes
        num_core_pixels = core.sum()
        penumbra = (mistar_array == 120) * 1
        num_penumbra_pixels = penumbra.sum()
        # get spacing and calculate volumes in mL
        x, y, z = mistar_img.GetSpacing()
        volume = (x * y * x)/1000
        core_volume = num_core_pixels * volume
        penumbra_volume = num_penumbra_pixels * volume
        results_df.loc[results_df.subject == subject, 'mistar_core'] = core_volume
        results_df.loc[results_df.subject == subject, 'mistar_penumbra'] = penumbra_volume
        results_df.loc[results_df.subject == subject, 'mistar_dice'] = dice_mistar

    results_df['mistar_mean_dice'] = results_df.mistar_dice.mean()
    results_df.to_csv(results_csv, index=None)


    fig, ax = plt.subplots(1, figsize=(8, 5))
    sm.graphics.mean_diff_plot(results_df.size_pred_ml, results_df.mistar_core, ax=ax)

    plt.savefig(results_folder + '/mistar_vs_dl_mean_diff_plot.png', dpi=300, facecolor='w', bbox_inches='tight', format='png')


if __name__ == '__main__':
    out_tag = 'best_model/stratify_size/att_unet_3_layers/without_atrophy/complete_occlusions'
    main(out_tag=out_tag)
