import sys
sys.path.append('/data/gpfs/projects/punim1086/ctp_project/MONAI/')
from monai_fns import *
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.handlers import EarlyStopHandler
from monai.handlers.utils import from_engine
from monai.utils import first, set_determinism
# from torchmetrics import Dice
from monai.visualize import GradCAM
from sklearn.metrics import classification_report
from monai.networks.nets import UNet, AttentionUnet
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    Invertd,
    GaussianSmoothd,
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RepeatChannelD,
    Resized,
    SaveImaged,
    ScaleIntensityd,
    ThresholdIntensityd,
    SplitDimd,
)

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import time
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# from numba import cuda
import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as f
from torch.optim import Adam
from models import *
from densenet import *


def define_zvalues(ct_img):
    z_min = int(ct_img.shape[2] * .25)
    z_max = int(ct_img.shape[2] * .85)

    steps = int((z_max - z_min) / 18)

    if steps == 0:
        z_min = 0
        z_max = ct_img.shape[2]
        steps = 1

    z = list(range(z_min, z_max))

    rem = int(len(z) / steps) - 18

    if rem < 0:
        add_on = [z[-1] for n in range(abs(rem))]
        z.extend(add_on)
    elif rem % 2 == 0:
        z_min = z_min + int(rem / 2 * steps) + 1
        z_max = z_max - int(rem / 2 * steps) + 1

    elif rem % 2 != 0:
        z_min = z_min + math.ceil(rem / 2)
        z_max = z_max - math.ceil(rem / 2) + 1

    z = list(range(z_min, z_max, steps))

    if len(z) == 19:
        z = z[1:]
    elif len(z) == 20:
        z = z[1:]
        z = z[:18]

    return z


def create_dwi_ctp_proba_map(dwi_ct_img,
                            gt,
                            proba_50,
                            proba_70,
                            proba_90,
                            savefile,
                            z,
                            ext='png',
                            save=False,
                            dpi=250):
    dwi_ct_img, gt, proba_50, proba_70, proba_90 = [np.rot90(im) for im in [dwi_ct_img, gt, proba_50, proba_70, proba_90]]
    dwi_ct_img, gt, proba_50, proba_70, proba_90 = [np.fliplr(im) for im in [dwi_ct_img, gt, proba_50, proba_70, proba_90]]
    proba_50_mask = proba_50 == 0
    proba_70_mask = proba_70 == 0
    proba_90_mask = proba_90 == 0
    masked_dwi = np.ma.array(dwi_ct_img, mask=~proba_50_mask)
    gt_mask = gt == 0
    masked_dwi_gt = np.ma.array(dwi_ct_img, mask=~gt_mask)
    proba_50 = np.where(proba_50 == 0, np.nan, proba_50)
    proba_70 = np.where(proba_70 == 0, np.nan, proba_70)
    proba_90 = np.where(proba_90 == 0, np.nan, proba_90)
    gt = np.where(gt == 0, np.nan, gt)


    fig, axs = plt.subplots(6, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.1, wspace=-0.3)
    axs = axs.ravel()
    for ax in axs:
        ax.axis("off")
    for i in range(6):
        print(i)

        axs[i].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        axs[i].imshow(gt[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)
        axs[i+6].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        axs[i+6].imshow(proba_50[:, :, z[i]], cmap='gnuplot',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
        axs[i+6].imshow(proba_70[:, :, z[i]], cmap='Wistia',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
        axs[i+6].imshow(proba_90[:, :, z[i]], cmap='bwr',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)

    if 12 > len(z):
        max2 = len(z)
    else:
        max2 = 12
    for i in range(6, max2):
        print(i)
        axs[i + 6].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        axs[i + 6].imshow(gt[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)
        axs[i+12].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        im = axs[i+12].imshow(proba_50[:, :, z[i]], cmap='gnuplot',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
        axs[i+12].imshow(proba_70[:, :, z[i]], cmap='Wistia',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
        axs[i+12].imshow(proba_90[:, :, z[i]], cmap='bwr',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
    if not 12 > len(z):
        if len(z) > 18:
            max3 = 18
        else:
            max3 = len(z)
        for i in range(12, max3):
            print(i)
            axs[i + 12].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                              interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
            axs[i + 12].imshow(gt[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)
            axs[i + 18].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                               interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
            im = axs[i + 18].imshow(proba_50[:, :, z[i]], cmap='gnuplot',
                                    interpolation='hanning', alpha=1, vmin=0, vmax=1)
            axs[i + 18].imshow(proba_70[:, :, z[i]], cmap='Wistia',
                               interpolation='hanning', alpha=1, vmin=0, vmax=1)
            axs[i + 18].imshow(proba_90[:, :, z[i]], cmap='bwr',
                               interpolation='hanning', alpha=1, vmin=0, vmax=1)

    if savefile:
        plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
        plt.close()



def main(notes=''):
    HOMEDIR = os.path.expanduser('~/')
    if os.path.exists(HOMEDIR + 'mediaflux/'):
        directory = HOMEDIR + 'mediaflux/data_freda/ctp_project/CTP_DL_Data/'
        ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])
        atlas_df = pd.read_excel(HOMEDIR + 'PycharmProjects/study_design/ATLAS_clinical_2023-02-14T1206.xlsx',
                                 sheet_name='Sheet1',
                                 header=[0],
                                 usecols=['INSPIRE ID', 'Occlusion severity (TIMI:0=complete occlusion, 3=normal)'])
    elif os.path.exists('Z:/data_freda'):
        directory = 'Z:/data_freda/ctp_project/CTP_DL_Data/'
        ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])
        atlas_df = pd.read_excel(HOMEDIR + 'PycharmProjects/study_design/ATLAS_clinical_2023-02-14T1206.xlsx',
                                 sheet_name='Sheet1',
                                 header=[0],
                                 usecols=['INSPIRE ID', 'Occlusion severity (TIMI:0=complete occlusion, 3=normal)'])
    elif os.path.exists('/data/gpfs/projects/punim1086/ctp_project'):
        directory = '/data/gpfs/projects/punim1086/ctp_project/CTP_DL_Data/'
        ctp_dl_df = pd.read_csv('/data/gpfs/projects/punim1086/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])
        atlas_df = pd.read_excel('/data/gpfs/projects/punim1086/study_design/ATLAS_clinical_2023-02-14T1206.xlsx',
                                 sheet_name='Sheet1',
                                 header=[0],
                                 usecols=['INSPIRE ID', 'Occlusion severity (TIMI:0=complete occlusion, 3=normal)'])
    elif os.path.exists('/media/mbcneuro'):
        directory = '/media/mbcneuro/CTP_DL_Data/'
        ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])
    elif os.path.exists('/media/fwerdiger'):
        directory = '/media/fwerdiger/Storage/CTP_DL_Data/'
        ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])
    elif os.path.exists('D:/ctp_project_data/CTP_DL_Data/'):
        directory = 'D:/ctp_project_data/CTP_DL_Data/'
        ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])

    # train/validation/test split

    data_dir = os.path.join(directory, 'DATA')
    image_paths = glob.glob(os.path.join(data_dir, 'images', '*'))
    image_paths.sort()
    mask_paths = glob.glob(os.path.join(data_dir, 'masks', '*'))
    mask_paths.sort()
    ncct_paths = glob.glob(os.path.join(data_dir, 'ncct', '*'))
    ncct_paths.sort()

    # filter out for TIMI = 0

    atlas_df['timi'] = atlas_df['Occlusion severity (TIMI:0=complete occlusion, 3=normal)'].apply(pd.to_numeric, errors='coerce')
    timi_df = ctp_dl_df.join(atlas_df.set_index('INSPIRE ID'), on='subject', how='left')
    incomplete_occlusions = timi_df[timi_df.apply(lambda x: x.timi > 0, axis=1)].drop_duplicates()
    incomplete_ids = incomplete_occlusions.dl_id.to_list()
    complete_occlusions = timi_df.drop(incomplete_occlusions.index)

    image_paths = [path for path in image_paths if not any(str(id).zfill(3) + '.nii.gz' in path for id in incomplete_ids)]
    mask_paths = [path for path in mask_paths if not any(str(id).zfill(3) + '.nii.gz' in path for id in incomplete_ids)]
    ncct_paths = [path for path in ncct_paths if not any(str(id).zfill(3) + '.nii.gz' in path for id in incomplete_ids)]

    assert len(image_paths) == len(mask_paths) == len(ncct_paths)

    random_state = 42
    # create column with size of lesion (in voxels)
    lesion_size = []
    for path in mask_paths:
        im = sitk.ReadImage(path)
        x, y, z = im.GetSpacing()
        voxel_size = (x * y * z)/1000
        label = sitk.LabelShapeStatisticsImageFilter()
        label.Execute(sitk.Cast(im, sitk.sitkUInt8))
        size = label.GetNumberOfPixels(1)
        lesion_size.append(voxel_size * size)

    # lesions less than 5 mL
    labels = (np.asarray(lesion_size) < 5) * 1
    complete_occlusions['size_labels'] = labels

    num_train = int(np.ceil(0.6 * len(labels)))
    num_validation = int(np.ceil(0.2 * len(labels)))
    num_test = len(labels) - (num_train + num_validation)

    train_id, test_id = train_test_split(complete_occlusions.dl_id.to_list(),
                                         train_size=num_train,
                                         test_size=num_test+num_validation,
                                         random_state=random_state,
                                         shuffle=True,
                                         stratify=labels)

    # get labels list that correspond with test id
    test_df = complete_occlusions[complete_occlusions.apply(lambda x: x['dl_id'] in test_id, axis=1)]
    test_labels = test_df.size_labels.to_list()
    test_id = test_df.dl_id.to_list()

    validation_id, test_id = train_test_split(test_id,
                                              train_size=num_validation,
                                              test_size=num_test,
                                              random_state=random_state,
                                              shuffle=True,
                                              stratify=test_labels)

    # HOME MANY TRAINING FILES ARE MANUALLY SEGMENTED
    train_df = complete_occlusions[complete_occlusions.apply(lambda x: x.dl_id in train_id, axis=1)]
    num_semi_train = len(train_df[train_df.apply(lambda x: x.segmentation_type == "semi_automated", axis=1)])
    num_small_train = len(train_df[train_df.apply(lambda x: x.size_labels == 1, axis=1)])

    val_df = complete_occlusions[complete_occlusions.apply(lambda x: x.dl_id in validation_id, axis=1)]
    num_semi_val = len(val_df[val_df.apply(lambda x: x.segmentation_type == "semi_automated", axis=1)])
    num_small_val = len(val_df[val_df.apply(lambda x: x.size_labels == 1, axis=1)])

    test_df = complete_occlusions[complete_occlusions.apply(lambda x: x.dl_id in test_id, axis=1)]
    num_semi_test = len(test_df[test_df.apply(lambda x: x.segmentation_type == "semi_automated", axis=1)])
    num_small_test = len(test_df[test_df.apply(lambda x: x.size_labels == 1, axis=1)])


    def make_dict(id):
        id = [str(num).zfill(3) for num in id]
        paths1 = [file for file in image_paths
                             if file.split('.nii.gz')[0].split('_')[-1] in id]
        paths2 = [file for file in ncct_paths if file.split('.nii.gz')[0].split('_')[-1] in id]
        paths3 = [file for file in mask_paths if file.split('.nii.gz')[0].split('_')[-1] in id]

        files_dict = [{"image": image_name, "ncct": ncct_name, "label": label_name} for image_name, ncct_name, label_name in zip(paths1, paths2, paths3)]

        return files_dict

    train_files = make_dict(train_id)
    val_files = make_dict(validation_id)
    test_files = make_dict(test_id)

    # model parameters
    max_epochs = 400
    image_size = [128]
    # feature order = ['DT', 'CBF', 'CBV', 'MTT', 'ncct', 'ncct_atrophy']
    features = ['DT', 'CBF', 'ncct']
    features_transform = ['image_' + string for string in [feature for feature in features
                                                           if "ncct" not in feature and "atrophy" not in feature]]
    if 'ncct' in features:
        features_transform += ['ncct_raw']
    if 'atrophy' in features:
        features_transform += ['ncct_atrophy']
        atrophy = True
    else:
        atrophy = False
    features_string = ''
    for feature in features:
        features_string += '_'
        features_string += feature
    patch_size = None
    batch_size = 2
    val_interval = 2
    out_tag = 'best_model/stratify_size/att_unet_3_layers/without_atrophy/complete_occlusions/'

    print(f"out_tag = {out_tag}")

    if not os.path.exists(directory + 'out_' + out_tag):
        os.makedirs(directory + 'out_' + out_tag)

    set_determinism(seed=42)

    transform_dir = os.path.join(directory, 'train', 'ncct_trans')
    if not os.path.exists(transform_dir):
        os.makedirs(transform_dir)


    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "ncct", "label"]),
            EnsureChannelFirstd(keys=["image", "ncct", "label"]),
            Resized(keys=["image", "ncct", "label"],
                    mode=['trilinear', 'trilinear', "nearest"],
                    align_corners=[True, True, None],
                    spatial_size=image_size*3),
            SplitDimd(keys="image", dim=0, keepdim=True,
                      output_postfixes=['DT', 'CBF', 'CBV', 'MTT']),
            RepeatChannelD(keys="ncct", repeats=2),
            SplitDimd(keys="ncct", dim=0, keepdim=True,
                      output_postfixes=['raw', 'atrophy']),
            ThresholdIntensityd(keys="ncct_atrophy", threshold=15, above=False),
            ThresholdIntensityd(keys="ncct_atrophy", threshold=0, above=True),
            GaussianSmoothd(keys="ncct_atrophy", sigma=1),
            ConcatItemsd(keys=features_transform, name="image", dim=0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=1.0),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "ncct", "label"]),
            EnsureChannelFirstd(keys=["image", "ncct", "label"]),
            Resized(keys=["image", "ncct", "label"],
                    mode=['trilinear', 'trilinear', "nearest"],
                    align_corners=[True, True, None],
                    spatial_size=image_size*3),
            SplitDimd(keys="image", dim=0, keepdim=True,
                      output_postfixes=["DT", "CBF", "CBV", "MTT"]),
            RepeatChannelD(keys="ncct", repeats=2),
            SplitDimd(keys="ncct", dim=0, keepdim=True,
                      output_postfixes=['raw', 'atrophy']),
            ThresholdIntensityd(keys="ncct_atrophy", threshold=15, above=False),
            ThresholdIntensityd(keys="ncct_atrophy", threshold=0, above=True),
            GaussianSmoothd(keys="ncct_atrophy", sigma=1),
            ConcatItemsd(keys=features_transform, name="image", dim=0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "ncct", "label"]),
            EnsureChannelFirstd(keys=["image", "ncct", "label"]),
            Resized(keys=["image", "ncct"],
                    mode=['trilinear', 'trilinear'],
                    align_corners=[True, True],
                    spatial_size=image_size*3),
            SplitDimd(keys="image", dim=0, keepdim=True,
                      output_postfixes=['DT', 'CBF', 'CBV', 'MTT']),
            RepeatChannelD(keys="ncct", repeats=2),
            SplitDimd(keys="ncct", dim=0, keepdim=True,
                      output_postfixes=['raw', 'atrophy']),
            ThresholdIntensityd(keys="ncct_atrophy", threshold=15, above=False),
            ThresholdIntensityd(keys="ncct_atrophy", threshold=0, above=True),
            GaussianSmoothd(keys="ncct_atrophy", sigma=1),
            ConcatItemsd(keys=features_transform, name="image", dim=0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    #train_dataset = CacheDataset(
    #   data=train_files,
    #    transform=train_transforms,
    #    cache_rate=1.0,
    #    num_workers=8)

    #val_dataset = CacheDataset(
    #    data=val_files,
    #    transform=val_transforms,
    #    cache_rate=1.0,
    #    num_workers=8)

    test_ds = CacheDataset(
        data=test_files,
        transform=test_transforms,
        cache_rate=1.0,
        num_workers=8
    )

    #train_loader = DataLoader(train_dataset,
    #                          batch_size=batch_size,
    #                          shuffle=True,
    #                          pin_memory=True)

    #val_loader = DataLoader(val_dataset,
    #                        batch_size=batch_size,
    #                        pin_memory=True)

    test_loader = DataLoader(test_ds,
                             batch_size=1,
                             pin_memory=True)

    # # sanity check to see everything is there
    s = 50
    data_example = test_ds[0]
    ch_in = data_example['image'].shape[0]
    # plt.figure("image", (18, 4))
    # for i in range(ch_in):
    #     plt.subplot(1, ch_in + 1, i + 1)
    #     plt.title(f"image channel {i}")
    #     plt.imshow(data_example["image"][i, :, :, s].detach().cpu(), cmap="jet")
    #     plt.axis('off')
    # # also visualize the 3 channels label corresponding to this image
    # print(f"label shape: {data_example['label'].shape}")
    # plt.subplot(1, 6, 6)
    # plt.title("label")
    # plt.imshow(data_example["label"][0, :, :, s].detach().cpu(), cmap="jet")
    # plt.axis('off')
    # plt.show()
    # plt.close()

    device = 'cuda'
    channels = (16, 32, 64)

    model = UNet(
        spatial_dims=3,
        in_channels=ch_in,
        out_channels=2,
        channels=channels,
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.2
    )
    model = DenseNetFCN(
        ch_in=ch_in,
        ch_out_init=36,
        num_classes=2,
        growth_rate=12,
        layers=(4, 4, 4, 4, 4),
        bottleneck=True,
        bottleneck_layer=4
    )
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=ch_in,
        out_channels=2,
        channels=channels,
        strides=(2, 2, 2),
    )
    # model = U_Net(ch_in, 2)
    # model = AttU_Net(ch_in, 2)

    # model = torch.nn.DataParallel(model)
    model.to(device)

    loss_function = DiceLoss(smooth_dr=1e-5,
                             smooth_nr=0,
                             to_onehot_y=True,
                             softmax=True,
                             include_background=False)
    loss_function = DiceCELoss(smooth_dr=1e-5,
                               smooth_nr=0,
                               to_onehot_y=True,
                               softmax=True,
                               include_background=False,
                               squared_pred=True,
                               lambda_dice=1,
                               lambda_ce=1)

    learning_rate = 1e-4
    optimizer = Adam(model.parameters(),
                     learning_rate,
                     weight_decay=1e-5)

    dice_metric = DiceMetric(include_background=False, reduction='mean')
    dice_metric_train = DiceMetric(include_background=False, reduction='mean')

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


    epoch_loss_values = []
    dice_metric_values = []
    dice_metric_values_train = []
    best_metric = -1
    best_metric_epoch = -1

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    start = time.time()
    model_path = 'best_metric_' + model._get_name() + '_' + str(max_epochs) + '_' + features_string +'.pth'

    # for epoch in range(max_epochs):
    #     print("-" * 10)
    #     print(f"epoch {epoch + 1}/{max_epochs}")
    #     epoch_loss = 0
    #     step = 0
    #     model.train()
    #     for batch_data in train_loader:
    #         step += 1
    #         inputs, labels = (
    #             batch_data["image"].to(device),
    #             batch_data["label"].to(device),
    #         )
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = loss_function(outputs, labels)
    #         loss.backward()
    #         epoch_loss += loss.item()
    #         optimizer.step()
    #     lr_scheduler.step()
    #     epoch_loss /= step
    #     epoch_loss_values.append(epoch_loss)
    #     print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    #
    #     if (epoch + 1) % val_interval == 0:
    #         model.eval()
    #         print("Evaluating...")
    #         with torch.no_grad():
    #             for val_data in val_loader:
    #                 val_inputs, val_labels = (
    #                     val_data["image"].to(device),
    #                     val_data["label"].to(device),
    #                 )
    #                 val_outputs = model(val_inputs)
    #
    #                 # compute metric for current iteration
    #                 # dice_metric_torch_macro(val_outputs, val_labels.long())
    #                 # now to for the MONAI dice metric
    #                 val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
    #                 val_labels = [post_label(i) for i in decollate_batch(val_labels)]
    #                 dice_metric(val_outputs, val_labels)
    #
    #             mean_dice = dice_metric.aggregate().item()
    #             dice_metric.reset()
    #             dice_metric_values.append(mean_dice)
    #
    #             # repeating the process for training data to check for overfitting
    #             for val_data in train_loader:
    #                 val_inputs, val_labels = (
    #                     val_data["image"].to(device),
    #                     val_data["label"].to(device),
    #                 )
    #                 val_outputs = model(val_inputs)
    #
    #                 # compute metric for current iteration
    #                 # dice_metric_torch_macro(val_outputs, val_labels.long())
    #                 # now to for the MONAI dice metric
    #                 val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
    #                 val_labels = [post_label(i) for i in decollate_batch(val_labels)]
    #                 dice_metric_train(val_outputs, val_labels)
    #
    #             mean_dice_train = dice_metric_train.aggregate().item()
    #             dice_metric_train.reset()
    #             dice_metric_values_train.append(mean_dice_train)
    #
    #             if mean_dice > best_metric:
    #                 best_metric = mean_dice
    #                 best_metric_epoch = epoch + 1
    #                 torch.save(model.state_dict(), os.path.join(
    #                     directory, 'out_' + out_tag, model_path))
    #                 print("saved new best metric model")
    #
    #             print(
    #                 f"current epoch: {epoch + 1} current mean dice: {mean_dice:.4f}"
    #                 f"\nbest mean dice: {best_metric:.4f} "
    #                 f"at epoch: {best_metric_epoch}"
    #             )
    #     del loss, outputs
    # end = time.time()
    # time_taken = end - start
    # print(f"Time taken: {round(time_taken, 0)} seconds")
    # time_taken_hours = time_taken/3600
    # time_taken_mins = np.ceil((time_taken/3600 - int(time_taken/3600)) * 60)
    # time_taken_hours = int(time_taken_hours)

    model_name = model._get_name()
    loss_name = loss_function._get_name()
    # with open(
    #         directory + 'out_' + out_tag + '/model_info_' + str(max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_' + features_string +'.txt', 'w') as myfile:
    #     myfile.write(f'Train dataset size: {len(train_files)}\n')
    #     myfile.write(f'Number semi-auto segmented: {num_semi_train}\n')
    #     myfile.write(f'Number of lesions under 5mL: {num_small_train}\n')
    #     myfile.write(f'Validation dataset size: {len(val_files)}\n')
    #     myfile.write(f'Number semi-auto segmented: {num_semi_val}\n')
    #     myfile.write(f'Number of lesions under 5mL: {num_small_val}\n')
    #     myfile.write(f'Test dataset size: {len(test_files)}\n')
    #     myfile.write(f'Number semi-auto segmented: {num_semi_test}\n')
    #     myfile.write(f'Number of lesions under 5mL: {num_small_test}\n')
    #     myfile.write(f'Intended number of features: {len(features)}\n')
    #     myfile.write(f'Actual number of features: {ch_in}\n')
    #     myfile.write('Features: ')
    #     myfile.write(features_string)
    #     myfile.write('\n')
    #     myfile.write(f'Model: {model_name}\n')
    #     myfile.write(f'Loss function: {loss_name}\n')
    #     myfile.write(f'Initial Learning Rate: {learning_rate}\n')
    #     myfile.write("Atrophy filter used? ")
    #     if atrophy:
    #         myfile.write("yes\n")
    #     else:
    #         myfile.write("no\n")
    #     myfile.write(f'Number of epochs: {max_epochs}\n')
    #     myfile.write(f'Batch size: {batch_size}\n')
    #     myfile.write(f'Image size: {image_size}\n')
    #     myfile.write(f'Patch size: {patch_size}\n')
    #     myfile.write(f'channels: {channels}\n')
    #     myfile.write(f'Validation interval: {val_interval}\n')
    #     myfile.write(f"Best metric: {best_metric:.4f}\n")
    #     myfile.write(f"Best metric epoch: {best_metric_epoch}\n")
    #     myfile.write(f"Time taken: {time_taken_hours} hours, {time_taken_mins} mins\n")
    #     myfile.write(notes)
    #
    #
    # # plot things
    # plt.figure("train", (12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("Average Loss per Epoch")
    # x = [i + 1 for i in range(len(epoch_loss_values))]
    # y = epoch_loss_values
    # plt.xlabel("epoch")
    # plt.plot(x, y)
    # plt.subplot(1, 2, 2)
    # plt.title("Mean Dice (Accuracy)")
    # x = [val_interval * (i + 1) for i in range(len(dice_metric_values))]
    # y = dice_metric_values
    # plt.xlabel("epoch")
    # plt.plot(x, y, 'b', label="Dice on validation data")
    # y = dice_metric_values_train
    # plt.plot(x, y, 'k', label="Dice on training data")
    # plt.legend(loc="center right")
    # plt.savefig(os.path.join(directory + 'out_' + out_tag,
    #                          'loss_plot_' + str(max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_' + features_string +'.png'),
    #             bbox_inches='tight', dpi=300, format='png')
    # plt.close()
    #
    # test

    pred_dir = os.path.join(directory + 'out_' + out_tag, "pred")
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    png_dir = os.path.join(directory + 'out_' + out_tag, "proba_pngs")
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    post_transforms = Compose([
        EnsureTyped(keys=["pred", "label"]),
        Invertd(
            keys=["pred", "proba"],
            transform=test_transforms,
            orig_keys=["image", "image"],
            meta_keys=["pred_meta_dict", "pred_meta_dict"],
            orig_meta_keys=["image_meta_dict", "image_meta_dict"],
            meta_key_postfix="meta_dict",
            nearest_interp=[False, False],
            to_tensor=[True, True],
        ),
        AsDiscreted(keys="label", to_onehot=2),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        SaveImaged(
            keys="proba",
            meta_keys="pred_meta_dict",
            output_dir=os.path.join(pred_dir, 'prob'),
            output_postfix="proba",
            resample=False,
            separate_folder=False),
        SaveImaged(
            keys="pred",
            meta_keys="pred_meta_dict",
            output_dir=pred_dir,
            output_postfix="seg",
            resample=False,
            separate_folder=False)
    ])

    loader = LoadImage(image_only=True)
    loader_meta = LoadImage(image_only=False)

    model.load_state_dict(torch.load(os.path.join(directory, 'out_' + out_tag, model_path)))

    model.eval()

    results = pd.DataFrame(columns=['id',
                                    'dice',
                                    'dice70',
                                    'dice90',
                                    'auc',
                                    'sensitivity',
                                    'specificity',
                                    'precision',
                                    'size',
                                    'size_pred',
                                    'px_x',
                                    'px_y',
                                    'px_z',
                                    'size_ml',
                                    'size_pred_ml'])
    results['id'] = [str(item).zfill(3) for item in test_id]
    # change ctp id dl id to string
    ctp_dl_df['dl_id'] = ctp_dl_df['dl_id'].apply(lambda row: str(row).zfill(3))
    ctp_dl_df.set_index('dl_id', inplace=True)

    from sklearn.metrics import f1_score, auc, recall_score, precision_score, roc_curve, confusion_matrix
    dice_metric = []
    dice_metric70 = []
    dice_metric90 = []
    sensitivities = []
    specificities = []
    # gts_flat = []
    # preds_flat = []

    # get hemisphere masks for each patients
    left_hemisphere_masks = glob.glob(directory + 'DATA/left_hemisphere_mask/*')
    right_hemisphere_masks = glob.glob(directory + 'DATA/right_hemisphere_mask/*')

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs = test_data["image"].to(device)

            test_data["pred"] = model(test_inputs)

            prob = f.softmax(test_data["pred"], dim=1)  # probability of infarct
            test_data["proba"] = prob

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

            test_output, test_label, test_image, test_proba = from_engine(
                ["pred", "label", "image", "proba"])(test_data)


            original_image = loader_meta(test_data[0]["image_meta_dict"]["filename_or_obj"])
            volx, voly, volz = original_image[1]['pixdim'][1:4]  # meta data
            pixel_vol = volx * voly * volz

            ground_truth = test_label[0][1].detach().numpy()
            prediction = (test_proba[0][1].detach().numpy() >= 0.5) * 1
            prediction_70 = (test_proba[0][1].detach().numpy() >= 0.7) * 1
            prediction_90 = (test_proba[0][1].detach().numpy() >= 0.9) * 1

            name = os.path.basename(
                test_data[0]["image_meta_dict"]["filename_or_obj"]).split('.nii.gz')[0].split('_')[1]
            subject = ctp_dl_df.loc[[name], "subject"].values[0]
            left_mask = [file for file in left_hemisphere_masks if name in file][0]
            right_mask = [file for file in right_hemisphere_masks if name in file][0]
            left_im, right_im = [loader(im) for im in [left_mask, right_mask]]
            left_np, right_np = [im.detach().numpy() for im in [left_im, right_im]]

            # find which hemisphere
            right_masked = right_np * prediction
            left_masked = left_np * prediction

            # see if there are any pixels in each corner
            hemisphere_mask = ''
            if np.count_nonzero(right_masked) > 0:
                hemisphere_mask = right_np.flatten()
            elif np.count_nonzero(left_masked) > 0:
                hemisphere_mask = left_np.flatten()

            gt_flat = ground_truth.flatten()
            # gts_flat.extend(gt_flat.astype(int))
            pred_flat = prediction.flatten()
            # preds_flat.extend(pred_flat.astype(int))
            pred70_flat = prediction_70.flatten()
            pred90_flat = prediction_90.flatten()
            dice_score = f1_score(gt_flat, pred_flat)
            dice_metric.append(dice_score)
            dice70 = f1_score(gt_flat, pred70_flat)
            dice_metric70.append(dice70)
            dice90 = f1_score(gt_flat, pred90_flat)
            dice_metric90.append(dice90)
            print(f"Dice score for image: {dice_score:.4f}")

            gt_flat = np.where((hemisphere_mask == 0), np.nan, gt_flat)
            core_flat = np.where(hemisphere_mask == 0, np.nan, core_flat)
            tp = len(np.where((gt_flat == 1) & (core_flat == 1))[0])
            fp = len(np.where((gt_flat == 0) & (core_flat == 1))[0])
            fn = len(np.where((gt_flat == 1) & (core_flat == 0))[0])
            tn = len(np.where((gt_flat == 0) & (core_flat == 0))[0])
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            # mask out nans and recalculate AUC
            fpr, tpr, threshold = roc_curve(gt_flat[np.where((gt_flat == 1) | (gt_flat == 0))],
                                            core_flat[np.where((core_flat == 1) | (core_flat == 0))])
            auc_score = auc(fpr, tpr)
            sensitivities.append(sensitivity)
            specificities.append(specificity)

            size = ground_truth.sum()
            size_ml = size * pixel_vol / 1000

            size_pred = prediction.sum()
            size_pred_ml = size_pred * pixel_vol / 1000


            try:
                dwi_img = glob.glob(os.path.join(directory, 'dwi_test/', subject + '*'))[0]
                dwi_img = loader(dwi_img)
                # spartan giving an error
                dwi_img = dwi_img.detach().numpy()

                save_loc = png_dir + '/' + subject + '_proba.png'
                create_dwi_ctp_proba_map(dwi_img, ground_truth, prediction, prediction_70, prediction_90, save_loc,
                                         define_zvalues(dwi_img), ext='png', save=True)
            except IndexError:
                print("no_dwi_image")

            results.loc[results.id == name, 'size'] = size
            results.loc[results.id == name, 'size_ml'] = size_ml
            results.loc[results.id == name, 'size_pred'] = size_pred
            results.loc[results.id == name, 'size_pred_ml'] = size_pred_ml
            results.loc[results.id == name, 'px_x'] = volx
            results.loc[results.id == name, 'px_y'] = voly
            results.loc[results.id == name, 'px_z'] = volz
            results.loc[results.id == name, 'dice'] = dice_score
            results.loc[results.id == name, 'dice70'] = dice70
            results.loc[results.id == name, 'dice90'] = dice90
            results.loc[results.id == name, 'auc'] = auc_score
            results.loc[results.id == name, 'sensitivity'] = sensitivity
            results.loc[results.id == name, 'specificity'] = specificity

        # aggregate the final mean dice result
        metric = np.mean(dice_metric)
        metric70 = np.mean(dice_metric70)
        metric90 = np.mean(dice_metric90)
        metric_recall = np.mean(sensitivities)
        metric_specificity = np.mean(specificities)
        # reset the status for next validation round
    print(f"Mean dice on test set: {metric:.4f}")
    results['mean_dice'] = metric
    results['mean_dice_70'] = metric70
    results['mean_dice_90'] = metric90
    results['mean_sensitvity'] = metric_recall
    results['mean_specificity'] = metric_specificity
    results_join = results.join(
        ctp_dl_df[~ctp_dl_df.index.duplicated(keep='first')],
        on='id',
        how='left')
    results_join.to_csv(directory + 'out_' + out_tag + '/results_' + str(max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_' + features_string + '.csv', index=False)

    # fpr, tpr, threshold = roc_curve(gts_flat, preds_flat)
    # roc_df = pd.DataFrame(np.asarray([gts_flat, preds_flat]).transpose(), columns=['ground_truth', 'prediction'])
    # roc_df.to_csv(directory + 'out_' + out_tag + '/roc_data_' + str(max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_' + features_string + '.csv', index=False)
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
    # plt.savefig(os.path.join(directory + 'out_' + out_tag,
    #                          'roc_plot_' + str(max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_' + features_string +'.png'),
    #             bbox_inches='tight', dpi=300, format='png')
    # plt.close()

if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    main(*sys.argv[1:])

