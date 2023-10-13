# this code is to compare MiStar map to my own prediction
import os

import numpy as np
import pandas as pd
import glob
import SimpleITK as sitk
from sklearn.metrics import f1_score, confusion_matrix, recall_score, roc_curve, auc
import statsmodels.api as sm
import matplotlib.pyplot as plt


def get_subject_results(subject, dl_id, gt_folder, atlas, directory):
    mistar_dir = atlas + '/ATLAS_database/' + subject + '/CT_baseline/CTP_baseline/mistar/'
    mistar_lesion = glob.glob(mistar_dir + '*' + 'Lesion.nii.gz')[0]
    mistar_img = sitk.ReadImage(mistar_lesion)
    mistar_array = sitk.GetArrayFromImage(mistar_img).astype(float)
    # get hemisphere mask
    left_mask = [file for file in glob.glob(directory + 'DATA/left_hemisphere_mask/*') if dl_id in file][0]
    right_mask = [file for file in glob.glob(directory + 'DATA/right_hemisphere_mask/*') if dl_id in file][0]
    left_im, right_im = [sitk.GetArrayFromImage(sitk.ReadImage(im)).astype(float)
                         for im in [left_mask, right_mask]]

    # find which hemisphere
    right_masked = right_im * mistar_array
    left_masked = left_im * mistar_array
    hemisphere_mask = ''
    counts_right = np.count_nonzero(right_masked)
    counts_left = np.count_nonzero(left_masked)
    if counts_right > counts_left:
        hemisphere_mask = right_im.ravel()
    elif counts_right < counts_left:
        hemisphere_mask = left_im.ravel()

    # get core and penumbra
    core = (mistar_array == 220) * 1
    core = np.asarray(core)
    # get gt
    gt = glob.glob(gt_folder + '/*' + dl_id + '*')[0]
    gt_array = sitk.GetArrayFromImage(sitk.ReadImage(gt))
    gt_flat = gt_array.ravel()
    core_flat = core.ravel()
    dice_mistar = f1_score(gt_flat, core_flat)
    tn, fp, fn, tp = confusion_matrix(gt_flat, core_flat).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    fpr, tpr, threshold = roc_curve(gt_flat, core_flat)
    old_roc_auc = auc(fpr, tpr)

    # get volumes
    num_core_pixels = core.sum()
    penumbra = (mistar_array == 120) * 1
    num_penumbra_pixels = penumbra.sum()
    # get spacing and calculate volumes in mL
    x, y, z = mistar_img.GetSpacing()
    volume = (x * y * x) / 1000
    core_volume = num_core_pixels * volume
    penumbra_volume = num_penumbra_pixels * volume


    gt_flat = np.where((hemisphere_mask==0), np.nan, gt_flat)
    core_flat = np.where(hemisphere_mask==0, np.nan, core_flat)
    tp = len(np.where((gt_flat==1) &(core_flat==1))[0])
    fp = len(np.where((gt_flat==0) &(core_flat==1))[0])
    fn = len(np.where((gt_flat==1) &(core_flat==0))[0])
    tn = len(np.where((gt_flat==0) &(core_flat==0))[0])
    if (tp == 0) and (fn == 0):
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    if (tp == 0) and (fp == 0):
        if fn == 0:
            ppv = 1
        else:
            ppv = 0
    else:
        ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    # mask out nans and recalculate AUC
    fpr, tpr, threshold = roc_curve(gt_flat[np.where((gt_flat == 1)|(gt_flat == 0))], core_flat[np.where((core_flat == 1)| (core_flat == 0))])
    roc_auc = auc(fpr, tpr)
    # gt_flat, core_flat = [int(arr) for arr in [gt_flat, core_flat]]
    return core_volume, penumbra_volume, dice_mistar, sensitivity, specificity, roc_auc, old_roc_auc, ppv, npv, gt_flat, core_flat


def main(out_tag):
    HOMEDIR = os.path.expanduser('~/')

    if os.path.exists(HOMEDIR + 'mediaflux/'):
        mediaflux = HOMEDIR + 'mediaflux/'
        directory = HOMEDIR + 'mediaflux/data_freda/ctp_project/CTP_DL_Data/'
        atlas = HOMEDIR + 'atlas/'

        ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])
        atlas_df = pd.read_excel(HOMEDIR + 'PycharmProjects/study_design/ATLAS_clinical_2023-06-09T0451.xlsx',
                                 sheet_name='Sheet1',
                                 header=[0],
                                 usecols=['INSPIRE ID', 'Occlusion severity (TIMI:0=complete occlusion, 3=normal)'])
    elif os.path.exists('Z:/data_freda'):
        mediaflux = 'Z:'
        atlas = 'Y:'
        directory = 'Z:/data_freda/ctp_project/CTP_DL_Data/'
        ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])
        atlas_df = pd.read_excel(HOMEDIR + 'PycharmProjects/study_design/ATLAS_clinical_2023-06-09T0451.xlsx',
                                 sheet_name='Sheet1',
                                 header=[0],
                                 usecols=['INSPIRE ID', 'Occlusion severity (TIMI:0=complete occlusion, 3=normal)'])

    print(f"out_tag = {out_tag}")

    results_folder = os.path.join(directory, 'out_' + out_tag)
    results_csv = glob.glob(results_folder + '/results*.csv')
    results_csv.sort()
    results_csv = results_csv[0]
    results_df = pd.read_csv(results_csv)
    results_df['mistar_core'] = ''
    results_df['mistar_penumbra'] = ''
    results_df['mistar_dice'] = ''
    results_df['mistar_sensitivity'] = ''
    results_df['mistar_specificity'] = ''
    results_df['mistar_auc'] = ''
    results_df['old_auc'] = ''

    gt_folder = os.path.join(directory, 'DATA', 'masks')
    gts_flat = []
    cores_flat = []
    voxel_ids = []
    # subject = results_df.subject.to_list()[1]
    for subject in results_df.subject.to_list():
        print("Running for {}".format(subject))
        dl_id = str(results_df.loc[results_df.subject == subject, 'id'].values[0]).zfill(3)
        results = get_subject_results(subject, dl_id, gt_folder, atlas, directory)
        core_volume, penumbra_volume, dice_mistar, sensitivity, specificity, roc_auc, old_auc, ppv, npv , _, _= results

        results_df.loc[results_df.subject == subject, 'mistar_core'] = core_volume
        results_df.loc[results_df.subject == subject, 'mistar_penumbra'] = penumbra_volume
        results_df.loc[results_df.subject == subject, 'mistar_dice'] = dice_mistar
        results_df.loc[results_df.subject == subject, 'mistar_sensitivity'] = sensitivity
        results_df.loc[results_df.subject == subject, 'mistar_specificity'] = specificity
        results_df.loc[results_df.subject == subject, 'mistar_ppv'] = ppv
        results_df.loc[results_df.subject == subject, 'mistar_npv'] = npv
        results_df.loc[results_df.subject == subject, 'mistar_auc'] = roc_auc
        results_df.loc[results_df.subject == subject, 'old_auc'] = old_auc

        gt_array = results[-2].tolist()
        core_array = results[-1].tolist()
        gts_flat.extend(gt_array)
        cores_flat.extend(core_array)

    results_df['mistar_mean_dice'] = results_df.mistar_dice.mean()
    results_df['mistar_mean_auc'] = results_df.mistar_auc.mean()
    results_df['mistar_old_mean_auc'] = results_df.old_auc.mean()
    results_df['mistar_mean_sensitivity'] = results_df.mistar_sensitivity.mean()
    results_df['mistar_mean_specificity'] = results_df.mistar_specificity.mean()

    results_df.to_csv(results_csv, index=None)

    # fig, ax = plt.subplots(1, figsize=(8, 5))
    # sm.graphics.mean_diff_plot(results_df.size_pred_ml, results_df.mistar_core, ax=ax)
    #
    # plt.savefig(results_folder + '/mistar_vs_dl_mean_diff_plot.png', dpi=300, facecolor='w', bbox_inches='tight', format='png')
    #
    # fpr, tpr, threshold = roc_curve(gts_flat, cores_flat)
    # roc_df = pd.DataFrame(np.array([gts_flat, cores_flat]).transpose(), columns=['gt', 'mistar_core'])
    # roc_df.to_csv(results_folder + '/mistar_roc_data.csv', index=False)
    #
    # roc_auc = auc(fpr, tpr)
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('Sensitivity')
    # plt.xlabel('1 - Specificity')
    # plt.savefig(os.path.join(results_folder + 'out_' + out_tag,
    #                          'mistar_roc_plot.png'),
    #             bbox_inches='tight', dpi=300, format='png')
    # plt.close()

if __name__ == '__main__':
    out_tag = 'best_model/stratify_size/att_unet_3_layers/without_atrophy/complete_occlusions/more_data_with_exclusions602020split/hemisphere'
    main(out_tag=out_tag)
