import glob
import os
import pandas as pd
import shutil

ctp_df = pd.read_csv(
            'C:/Users/fwerdiger/PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
            usecols=['subject', 'dl_id'])

test_df = ctp_df[ctp_df.apply(lambda x: 'test' in x.dl_id, axis=1)]

subjects = test_df.subject.to_list()
out_dir = 'D:/ctp_project_data/CTP_DL_Data/dwi_test/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for subject in subjects:
    dwi_img = glob.glob('Z:/INSPIRE_database/' + subject + '/CT_baseline/CTP_baseline/transform-DWI_followup/*__Warped.nii.gz')[0]
    shutil.copy(dwi_img, os.path.join(out_dir, subject + '_dwi.nii.gz'))
