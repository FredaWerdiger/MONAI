import os
import SimpleITK as sitk

def get_semi_dataset():
    HOMEDIR = os.path.expanduser('~/')

    mediaflux = HOMEDIR + 'mediaflux'

    if os.path.exists(mediaflux):
        semi_data = os.path.join(mediaflux, 'data_freda/ctp_project/CTP_DL_Data/no_seg')
    elif os.path.exists('Z:/data_freda'):
        semi_data = os.path.join('Z:', 'data_freda/ctp_project/CTP_DL_Data/no_seg')
    elif os.path.exists('/data/gpfs/projects/punim1086/ctp_project'):
        semi_data = '/data/gpfs/projects/punim1086/ctp_project/CTP_DL_Data/no_seg'

    masks_semi = [file for file in os.listdir(os.path.join(semi_data, 'masks_semi'))
                 if ('exclude' not in file and 'funny' not in file)]
    masks_semi.sort()

    subjects_semi = ['INSP_' + file.split('_')[2].split('.nii')[0] for file in masks_semi]
    # for subject, file in zip(subjects_semi, masks_semi):
    #     os.rename(os.path.join(semi_data, 'mask_semi', file),
    #               os.path.join(semi_data, 'mask_semi', 'mask_' + subject + '.nii.gz'))
    # exclude cases with two dwi channels - from CN13 or CN16
    subjects_semi = [subject for subject in subjects_semi if
                     ('CN13' not in subject and 'CN16' not in subject)]
    images_semi = [os.path.join(semi_data, 'images', file)
                   for file in os.listdir(os.path.join(semi_data, 'images'))
                   if 'INSP_' + file.split('_')[2].split('.nii')[0] in subjects_semi]
    images_semi.sort()
    subjects_semi.sort()

    # go through and make sure there no single-channel images
    subjects_in = []
    for subject, image in zip(subjects_semi, images_semi):
        im = sitk.ReadImage(image)
        sz = im.GetSize()
        if len(sz) == 4:
            if sz[-1] == 2:
                subjects_in.append(subject)
    images_semi = [os.path.join(semi_data, 'images', file)
                   for file in os.listdir(os.path.join(semi_data, 'images'))
                   if 'INSP_' + file.split('_')[2].split('.nii')[0] in subjects_in]
    masks_semi = [os.path.join(semi_data, 'masks_semi', file)
                   for file in os.listdir(os.path.join(semi_data, 'masks_semi'))
                   if 'INSP_' + file.split('_')[2].split('.nii')[0] in subjects_in]

    images_semi.sort()
    masks_semi.sort()

    semi_dict = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images_semi, masks_semi)
    ]
    return semi_dict

def get_corrections():
    HOMEDIR = os.path.expanduser('~/')

    mediaflux = 'Z:/'
    if os.path.exists(HOMEDIR + 'mediaflux/'):
        mediaflux = HOMEDIR + 'mediaflux/'

    if not os.path.exists(mediaflux):

        semi_data = '/data/gpfs/projects/punim1086/ctp_project/DWI_Training_Data/no_seg'

        masks_semi = [file for file in os.listdir(os.path.join(semi_data, 'densenet_corrections'))
                     if ('seg' in file and 'INSP_CN230483' not in file and 'INSP_CN230496' not in file)]
        masks_semi.sort()


        subjects_semi = ['INSP_' + file.split('_')[2] for file in masks_semi]
        # for subject, file in zip(subjects_semi, masks_semi):
        #     os.rename(os.path.join(semi_data, 'mask_semi', file),
        #               os.path.join(semi_data, 'mask_semi', 'mask_' + subject + '.nii.gz'))

        images_semi = [os.path.join(semi_data, 'densenet_corrections', 'image_' + subject + '.nii.gz')
                       for subject in subjects_semi]
        masks_semi = [os.path.join(semi_data, 'densenet_corrections', file) for file in masks_semi]
    else:
        semi_data = mediaflux + 'data_freda/dl_seg_check/'
        masks_semi = [file for file in os.listdir(semi_data)
                      if ('seg' in file and 'INSP_CN230483' not in file and 'INSP_CN230496' not in file)]
        masks_semi.sort()

        subjects_semi = ['INSP_' + file.split('_')[2] for file in masks_semi]
        # for subject, file in zip(subjects_semi, masks_semi):
        #     os.rename(os.path.join(semi_data, 'mask_semi', file),
        #               os.path.join(semi_data, 'mask_semi', 'mask_' + subject + '.nii.gz'))

        images_semi = [os.path.join(semi_data, 'image_' + subject + '.nii.gz')
                       for subject in subjects_semi]
        masks_semi = [os.path.join(semi_data, file) for file in masks_semi]
    semi_dict = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images_semi, masks_semi)
    ]
    return semi_dict

