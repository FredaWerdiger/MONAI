import glob
import os
import pandas as pd
import shutil

ctp_df = pd.read_csv(
            '../study_design/study_lists/data_for_ctp_dl.csv',
            usecols=['subject', 'dl_id'])
HOMEDIR = os.path.expanduser('~/')
mediaflux = HOMEDIR + 'mediaflux/'
test_df = pd.read_csv(
    mediaflux + 'data_freda/ctp_project/CTP_DL_Data/out_best_model/stratify_size/att_unet_3_layers/without_atrophy/complete_occlusions/upsample/results_400_epoch_AttentionUnet_DiceCELoss__DT_CBF_ncct.csv',
usecols=['subject', 'id'])

# test_df = ctp_df[ctp_df.apply(lambda x: 'test' in x.dl_id, axis=1)]

subjects = test_df.subject.to_list()
out_dir = mediaflux + 'data_freda/ctp_project/CTP_DL_Data/dwi_test'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for subject in subjects:
    if not os.path.exists(os.path.join(out_dir, subject + '_dwi.nii.gz')):
        if not subject == 'INSP_AU250244':
            print('Running for {}'.format(subject))
            dwi_img = glob.glob('Z:/INSPIRE_database/' + subject + '/CT_baseline/CTP_baseline/transform-DWI_followup/*__Warped.nii.gz')[0]
            shutil.copy(dwi_img, os.path.join(out_dir, subject + '_dwi.nii.gz'))
