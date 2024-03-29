import os
import math
import pathlib
import sys

sys.path.append('/data/gpfs/projects/punim1086/ctp_project/MONAI')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
import glob
from monai.data import Dataset, DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet, UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    Invertd,
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    Resized,
    SaveImaged,

)
# from torchmetrics import Dice
import torch
import torch.nn.functional as f
from monai_fns import *
from densenet import *
import SimpleITK as sitk
from sklearn.model_selection import train_test_split


def define_dvalues(dwi_img):
    steps = int(dwi_img.shape[2] / 18)
    rem = int(dwi_img.shape[2] / steps) - 18

    if rem == 0:
        d_min = 0
        d_max = dwi_img.shape[2]
    elif rem % 2 == 0:
        d_min = 0 + int(rem / 2 * steps) + 1
        d_max = dwi_img.shape[2] - int(rem / 2 * steps) + 1

    elif rem % 2 != 0:
        d_min = 0 + math.ceil(rem * steps / 2)
        d_max = dwi_img.shape[2] - math.ceil(rem / 2 * steps) + 1

    d = range(d_min, d_max, steps)

    if len(d) == 19:
        d = d[1:]
    return d


def define_dvalues_big(dwi_img):
    dwi_img_small = dwi_img[10:120]
    steps = int(dwi_img_small.shape[0] / 18)
    rem = int(dwi_img_small.shape[0] / steps) - 18

    if rem % 2 == 0:
        d_min = 0 + int(rem / 2 * steps) + 1
        d_max = dwi_img_small.shape[0] - int(rem / 2 * steps)

    elif rem % 2 != 0:
        d_min = 0 + math.ceil(rem * steps / 2)
        d_max = dwi_img_small.shape[0] - math.ceil(rem / 2 * steps)

    d = range(d_min + 10, d_max + 10, steps)

    if len(d) == 19:
        d = range(d_min + steps + 10, d_max + 10, steps)
    return d


def create_mrlesion_img(dwi_img, dwi_lesion_img, savefile, d, ext='png', dpi=250):
    dwi_lesion_img = np.rot90(dwi_lesion_img)
    dwi_img = np.rot90(dwi_img)
    dwi_img, dwi_lesion_img = [np.fliplr(img) for img in [dwi_img, dwi_lesion_img]]

    mask = dwi_lesion_img < 1
    masked_im = np.ma.array(dwi_img, mask=~mask)

    fig, axs = plt.subplots(3, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.6, wspace=-0.1)

    axs = axs.ravel()

    for i in range(len(d)):
        axs[i].imshow(dwi_img[:, :, d[i]], cmap='gray', interpolation='hanning', vmin=0, vmax=300)
        axs[i].imshow(dwi_lesion_img[:, :, d[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=-2, vmax=1)
        axs[i].imshow(masked_im[:, :, d[i]], cmap='gray', interpolation='hanning', alpha=1, vmin=0, vmax=300)
        axs[i].axis('off')
    # plt.show()
    plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
    plt.close()


def create_mr_img(dwi_img, savefile, d, ext='png', dpi=250):
    dwi_img = np.rot90(dwi_img)
    fig, axs = plt.subplots(3, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.6, wspace=-0.1)
    axs = axs.ravel()
    for i in range(len(d)):
        axs[i].imshow(dwi_img[:, :, d[i]], cmap='gray', interpolation='hanning', vmin=0, vmax=300)
        axs[i].axis('off')
    # plt.show()
    plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
    plt.close()


def create_mr_big_img(dwi_img, savefile, d, ext='png', dpi=250):
    dwi_img = np.rot90(dwi_img)
    fig, axs = plt.subplots(3, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.6, wspace=-0.1)
    axs = axs.ravel()
    for i in range(len(d)):
        axs[i].imshow(dwi_img[:, :, d[i]], cmap='gray')
        axs[i].axis('off')
    # plt.show()
    plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
    plt.close()


def create_adc_img(dwi_img, savefile, d, ext='png', dpi=250):
    dwi_img = np.rot90(dwi_img)
    fig, axs = plt.subplots(3, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.6, wspace=-0.1)
    axs = axs.ravel()
    for i in range(len(d)):
        axs[i].imshow(dwi_img[:, :, d[i]], cmap='gray', interpolation='hanning', vmin=dwi_img.min(), vmax=dwi_img.max())
        axs[i].axis('off')
    # plt.show()
    plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
    plt.close()


def create_paper_img(dwi_img, gt, pred, savefile, d, ext='png', dpi=250):
    dwi_img, gt, pred = [np.rot90(im) for im in [dwi_img, gt, pred]]
    lesions = gt + pred
    mask = lesions == 0
    masked_img = np.ma.array(dwi_img, mask=~mask)

    false_neg = np.ma.masked_where(
        np.logical_and(pred == 0, gt == 1), gt) * gt
    true_pos = np.ma.masked_where(np.logical_and(pred == 1, gt == 1),
                                  gt) * gt
    false_pos = np.ma.masked_where(
        np.logical_and(pred == 1, gt == 0), pred) * pred

    fig, axs = plt.subplots(3, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.6, wspace=-0.1)
    axs = axs.ravel()
    for i in range(len(d)):
        axs[i].imshow(false_neg[:, :, d[i]], cmap='gist_rainbow', vmin=0, vmax=1)
        axs[i].imshow(false_pos[:, :, d[i]], cmap='brg', vmin=0, vmax=1)
        # axs[i].imshow(true_pos[:,:,d[i]], cmap='tab10', vmin=0, vmax=1)
        axs[i].imshow(masked_img[:, :, d[i]], cmap='gray', interpolation='hanning', vmin=0, vmax=300)
        axs[i].axis('off')
    # plt.show()
    plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
    plt.close()


def create_overviewhtml(subject_id, df, outdir):
    '''
    Function that creates a HTML file with clinical information and imaging.
    '''
    templateLoader = FileSystemLoader(
        searchpath='./')
    templateEnv = Environment(loader=templateLoader)
    TEMPLATE_FILE = "template.html"
    template = templateEnv.get_template(TEMPLATE_FILE)

    savefolder = outdir + 'htmls'
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    ptData = df.set_index('id').loc[subject_id, :]
    treat = ptData[
        'Treatment type(0=no treatment,1=iv only, 2=IA only, 3= Both ia +iv, 4=iv only IA planned but not delivery,5=no information)']
    if treat == 0:
        treat = "No treatment"
    elif treat == 1:
        treat = "IV only"
    elif treat == 2:
        treat = "IA only"
    elif treat == 3:
        treat = "IV and IA"

    small = ptData['small lesion']
    small = 'yes' if small == 1 else 'no'
    thickness = ptData['dwi_slice_thickness']
    make = str(ptData['dwi_Manufacturer']) + ' ' + str(ptData['dwi_Model'])
    day = ptData['day']
    cause = ptData['Stroke Mechanism']

    output = template.render(subject_id=subject_id,
                             inspire_id=ptData['subject'],
                             dice=round(ptData['dice'], 4),
                             folder=outdir + 'images',
                             treatment=treat,
                             make=make,
                             thickness=thickness,
                             small=small,
                             day=day,
                             cause=cause
                             )
    # populate template

    # save html file

    with open(os.path.join(savefolder, subject_id + '.html'), 'w') as f:
        f.write(output)


def make_dict(root, string):
    images = sorted(
        glob.glob(os.path.join(root, string, 'images', '*.nii.gz'))
    )
    labels = sorted(
        glob.glob(os.path.join(root, string, 'masks', '*.nii.gz'))
    )
    return [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images, labels)
    ]


def main(directory, ctp_df, model_path, out_tag, acute, follow_up, isles, ddp=False):
    # test on external data
    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            Resized(keys="image",
                    mode='trilinear',
                    align_corners=True,
                    spatial_size=(128, 128, 128)),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
            # SaveImaged(keys="image", output_dir=root_dir + "out", output_postfix="transform", resample=False)
        ]
    )

    if follow_up:
        test_files = BuildDataset(directory, 'no_seg/test_cases').images_dict
    if acute:
        test_files = BuildDataset(directory, 'test').images_dict
    if isles:
        test_files = BuildDataset(directory, 'ISLES22').images_dict
        lesion_size = []
        ids = []
        for patient in test_files:
            path = patient['label']
            id = path.split('.nii.gz')[0].split('_')[-1]
            ids.append(id)
            im = sitk.ReadImage(path)
            x, y, z = im.GetSpacing()
            voxel_size = (x * y * z) / 1000
            label = sitk.LabelShapeStatisticsImageFilter()
            label.Execute(sitk.Cast(im, sitk.sitkUInt8))
            try:
                size = label.GetNumberOfPixels(1)
            except RuntimeError:
                size = 0
            lesion_size.append(voxel_size * size)

        # lesions less than 5 mL
        labels = (np.asarray(lesion_size) < 5) * 1
        isles_df = pd.DataFrame(labels, columns=['small_lesion'], index=ids)
        isles_df['id'] = isles_df.index
        isles_df['big_lesion'] = (np.asarray(lesion_size) > 70) * 1
        _, test_ids = train_test_split(ids,
                                      train_size=230,
                                      test_size=20,
                                      random_state=371, # three big and three small
                                      shuffle=True)
        isles_test = isles_df[isles_df.apply(lambda x: x.id in test_ids, axis=1)]

        test_files = [file for file in test_files if any(id in file['label'] for id in test_ids)]

    test_ds = Dataset(
        data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)

    post_transforms = Compose([
        EnsureTyped(keys=["pred", "label"]),
        EnsureChannelFirstd(keys="label"),
        Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        AsDiscreted(keys="label", to_onehot=2),
        # SaveImaged(keys="pred",
        #            meta_keys="pred_meta_dict",
        #            output_dir=root_dir + "out_" + out_tag + '/pred',
        #            output_postfix="pred", resample=False,
        #            separate_folder=False),
    ])

    if not os.path.exists(directory + "out_" + out_tag + '/pred'):
        os.makedirs(directory + "out_" + out_tag + '/pred')

    # removing sync on step as we are running on master node
    # dice_metric = Dice(ignore_index=0)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    loader = LoadImage(image_only=False)
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2)
    ).to(device)

    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    model = DenseNetFCN(
      ch_in=2,
       ch_out_init=48,
       num_classes=2,
       growth_rate=16,
       layers=(4, 5, 7, 10, 12),
       bottleneck=True,
       bottleneck_layer=15
    ).to(device)

    if ddp:
        model.load_state_dict(ddp_state_dict(model_path))
    else:
        model.load_state_dict(torch.load(model_path))
    model.eval()

    results = pd.DataFrame(columns=['id', 'dice', 'size_ml', 'size_pred_ml'])
    if acute:
        results['id'] = ['test_' + str(item).zfill(3) for item in range(1, len(test_loader) + 1)]
    if follow_up:
        results['id'] = ['no_seg_' + file['image'].split('.nii.gz')[0].split('_')[-1] for file in test_files]
    if isles:
        results['id'] = ['isles_' + id for id in test_ids] # included isles images
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs = test_data["image"].to(device)
            test_data["pred"] = model(test_inputs)
            # roi_size = (64, 64, 64)
            # sw_batch_size = 4
            # test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

            test_output, test_label, test_image = from_engine(["pred", "label", "image"])(test_data)

            a = dice_metric(y_pred=test_output, y=test_label)

            dice_score = round(a.item(), 4)
            print(f"Dice score for image: {dice_score:.4f}")

            # get original image, and normalize it so we can see the normalized image
            # this is both channels
            original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])
            volx, voly, volz = original_image[1]['pixdim'][1:4]  # meta data
            pixel_vol = volx * voly * volz

            original_image = original_image[0]  # image data
            original_adc = original_image[:, :, :, 1]
            original_image = original_image[:, :, :, 0]
            ground_truth = test_label[0][1].detach().numpy()
            prediction = test_output[0][1].detach().numpy()
            transformed_image = test_inputs[0][0].detach().cpu().numpy()
            size = ground_truth.sum()
            size_ml = size * pixel_vol / 1000
            size_pred = prediction.sum()
            size_pred_ml = size_pred * pixel_vol / 1000
            # for acute test set
            if acute:
                name = "test_" + os.path.basename(
                    test_data[0]["image_meta_dict"]["filename_or_obj"]).split('.nii.gz')[0].split('_')[-1]
            # for follow-up test set
            if follow_up:
                name = "no_seg_" + os.path.basename(
                    test_data[0]["image_meta_dict"]["filename_or_obj"]).split('.nii.gz')[0].split('_')[-1]
            if isles:
                name = "isles_" + os.path.basename(
                    test_data[0]["image_meta_dict"]["filename_or_obj"]).split('.nii.gz')[0].split('_')[-1]
            save_loc = directory + "out_" + out_tag + "/images/" + name + "_"

            if not os.path.exists(directory + "out_" + out_tag + "/images/"):
                os.makedirs(directory + "out_" + out_tag + "/images/")

            # create_paper_img(
            #     original_image,
            #     ground_truth,
            #     prediction,
            #     save_loc + "paper.png",
            #     define_dvalues(original_image),
            #     'png',
            #     dpi=300
            # )
            # create_mr_img(
            #     original_image,
            #     save_loc + "dwi.png",
            #     define_dvalues(original_image),
            #     'png',
            #     dpi=300)
            # create_adc_img(
            #     original_adc,
            #     save_loc + "adc.png",
            #     define_dvalues(original_image),
            #     'png',
            #     dpi=300)
            #
            # [create_mrlesion_img(
            #     original_image,
            #     im,
            #     save_loc + name + '.png',
            #     define_dvalues(original_image),
            #     'png',
            #     dpi=300) for im, name in zip([prediction, ground_truth], ["pred", "truth"])]
            #
            # create_mr_big_img(transformed_image,
            #                   save_loc + "dwi_tran.png",
            #                   define_dvalues_big(transformed_image),
            #                   'png',
            #                   dpi=300)


            results.loc[results.id == name, 'size'] = size
            results.loc[results.id == name, 'size_ml'] = size_ml
            results.loc[results.id == name, 'size_pred_ml'] = size_pred_ml
            results.loc[results.id == name, 'dice'] = dice_score

        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

    print(f"Mean dice on test set: {metric:.4f}")

    results['mean_dice'] = metric

    results_join = results.join(
        ctp_df[~ctp_df.index.duplicated(keep='first')],
        on='id',
        how='left')
    print(results)
    results_join.to_csv(directory + 'out_' + out_tag + '/results.csv', index=False)

    # for sub in results_join['id']:
    #     create_overviewhtml(sub, results_join, root_dir + 'out_' + out_tag + '/')


if __name__ == '__main__':
    HOMEDIR = os.path.expanduser("~/")
    if os.path.exists(HOMEDIR + 'mediaflux/'):
        directory = HOMEDIR + 'mediaflux/data_freda/ctp_project/DWI_Training_Data/'
        ctp_df = pd.read_csv(
            '/home/unimelb.edu.au/fwerdiger/PycharmProjects/study_design/study_lists/dwi_segmentation_paper_patients.csv',
            index_col='dl_id'
        )
        windows = False
    elif os.path.exists('Z:'):
        directory = 'Z:/data_freda/ctp_project/DWI_Training_Data/'
        ctp_df = pd.read_csv(
            HOMEDIR + 'PycharmProjects/study_design/study_lists/dwi_segmentation_paper_patients.csv',
            index_col='dl_id')
        windows = True
    elif os.path.exists('/media/mbcneuro'):
        directory = '/media/mbcneuro/DWI_Training_Data/'
        ctp_df = pd.read_csv(
            '/home/mbcneuro/PycharmProjects/study_design/study_lists/dwi_segmentation_paper_patients.csv',
            index_col='dl_id'
        )
        windows = False
    elif os.path.exists('D:'):
        directory = 'D:/ctp_project_data/DWI_Training_Data/'
        ctp_df = pd.read_csv(
            'C:/Users/fwerdiger/PycharmProjects/study_design/study_lists/dwi_segmentation_paper_patients.csv',
            index_col='dl_id')
    else:
        directory = '/data/gpfs/projects/punim1086/ctp_project/DWI_Training_Data/'
        ctp_df = pd.read_csv(
            '/data/gpfs/projects/punim1086/study_design/study_lists/dwi_segmentation_paper_patients.csv',
            index_col='dl_id')

    model_path = directory + 'out_densenetFCN_batch1/learning_rate_1e4/without_extra_data/best_metric_model600.pth'
    out_tag = 'densenetFCN_batch1/learning_rate_1e4/without_extra_data/isles_test_set'
    main(directory, ctp_df, model_path, out_tag, acute=False, follow_up=False, isles=True, ddp=False)
