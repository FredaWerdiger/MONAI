# this code is to get a probability map out instead of a binary mask
# currently in piece
import sys
sys.path.append('/data/gpfs/projects/punim1086/ctp_project/MONAI/')
from monai_fns import *
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from jinja2 import Environment, FileSystemLoader
import glob
# from collections import OrderedDict
from monai.data import Dataset, DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet, UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    GaussianSmoothd,
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    Resized,
    SaveImage,
    SaveImaged,
    SplitDimd,
    ThresholdIntensityd,
)
import torch
import torch.nn.functional as f
from monai.utils import (
    BlendMode,
    PytorchPadMode
)
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

def create_dwi_ctp_proba_image(dwi_ct_img,
                               gt,
                               proba,
                               savefile,
                               z,
                               ext='png',
                               save=False,
                               dpi=250):
    dwi_ct_img, gt, proba = [np.rot90(im) for im in [dwi_ct_img, gt, proba]]
    dwi_ct_img, gt, proba = [np.fliplr(im) for im in [dwi_ct_img, gt, proba]]
    proba_mask = proba == 0
    masked_dwi = np.ma.array(dwi_ct_img, mask=~proba_mask)
    gt_mask = gt == 0
    masked_dwi_gt = np.ma.array(dwi_ct_img, mask=~gt_mask)

    fig, axs = plt.subplots(6, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.1, wspace=-0.3)
    axs = axs.ravel()
    for ax in axs:
        ax.axis("off")
    for i in range(6):
        print(i)

        axs[i].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        axs[i].imshow(gt[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.2)
        axs[i].imshow(masked_dwi_gt[:, :, z[i]], cmap='gray',
                      interpolation='hanning', alpha=1, vmin=10, vmax=dwi_ct_img.max())
        axs[i+6].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        axs[i+6].imshow(proba[:, :, z[i]], cmap='YlOrRd',
                      interpolation='hanning', alpha=0.6, vmin=-0.3, vmax=1.2)
        axs[i+6].imshow(masked_dwi[:, :, z[i]],
                      cmap='gray', interpolation='hanning',
                      alpha=1, vmin=10, vmax=dwi_ct_img.max())
    if 12 > len(z):
        max2 = len(z)
    else:
        max2 = 12
    for i in range(6, max2):
        print(i)
        axs[i + 6].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        axs[i + 6].imshow(gt[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.2)
        axs[i + 6].imshow(masked_dwi_gt[:, :, z[i]], cmap='gray',
                      interpolation='hanning', alpha=1, vmin=10, vmax=dwi_ct_img.max())
        axs[i + 12].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                          interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        im = axs[i + 12].imshow(proba[:, :, z[i]], cmap='YlOrRd',
                          interpolation='hanning', alpha=0.6, vmin=-0.3, vmax=1.2)
        axs[i + 12].imshow(masked_dwi[:, :, z[i]],
                          cmap='gray', interpolation='hanning',
                          alpha=1, vmin=10, vmax=dwi_ct_img.max())
    if not 12 > len(z):
        if len(z) > 18:
            max3 = 18
        else:
            max3 = len(z)
        for i in range(12, max3):
            print(i)
            axs[i + 12].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                          interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
            axs[i + 12].imshow(gt[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.2)
            axs[i + 12].imshow(masked_dwi_gt[:, :, z[i]], cmap='gray',
                   interpolation='hanning', alpha=1, vmin=10, vmax=dwi_ct_img.max())
            axs[i + 18].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                              interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
            axs[i + 18].imshow(proba[:, :, z[i]], cmap='YlOrRd',
                              interpolation='hanning', alpha=0.6, vmin=-0.3, vmax=1.2)
            axs[i + 18].imshow(masked_dwi[:, :, z[i]],
                              cmap='gray', interpolation='hanning',
                              alpha=1, vmin=10, vmax=dwi_ct_img.max())
    cbar = plt.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, boundaries=[0,0.5, 1])
    cbar.set_ticks(np.arange(0, 1.5, 0.5))
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.set_ylabel('Probability of Infarct', rotation=270)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    cbar.outline.set_edgecolor('white')
    plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
    plt.close()


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



def main(directory, ctp_df, ddp=False):


    out_tag = 'best_model/AttUNet'
    model_path = directory + 'out_' + out_tag + '/best_metric_AttU_Net_400_15HU_DT_CBF_ncct.pth'

    HU = 15
    image_size = [128]

    # feature order = ['DT', 'CBF', 'CBV', 'MTT', 'ncct']
    features = ['DT', 'CBF', 'ncct']
    features_transform = ['image_' + string for string in [feature for feature in features if "ncct" not in feature]]
    if 'ncct' in features:
        features_transform += ['ncct']
    features_string = ''
    for feature in features:
        features_string += '_'
        features_string += feature

    png_dir = os.path.join(directory + 'out_' + out_tag, "proba_pngs")
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    pred_dir = os.path.join(directory + 'out_' + out_tag, "pred")
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    atrophy_transforms = [
        ThresholdIntensityd(keys="ncct", threshold=HU, above=False),
        ThresholdIntensityd(keys="ncct", threshold=0, above=True),
        GaussianSmoothd(keys="ncct", sigma=1)
    ]

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
            *atrophy_transforms,
            ConcatItemsd(keys=features_transform, name="image", dim=0),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    test_files = BuildDataset(directory, 'test').ncct_dict

    test_ds = Dataset(
        data=test_files, transform=test_transforms)

    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

    data_example = test_ds[0]
    ch_in = data_example['image'].shape[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # test on external data
    channels = (16, 32, 64)
    model = UNet(
        spatial_dims=3,
        in_channels=ch_in,
        out_channels=2,
        channels=channels,
        strides=(2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.2
    ).to(device)
    model = DenseNetFCN(
        ch_in=ch_in,
        ch_out_init=36,
        num_classes=2,
        growth_rate=12,
        layers=(4, 4, 4, 4, 4),
        bottleneck=True,
        bottleneck_layer=4
    ).to(device)
    model = U_Net(ch_in, 2).to(device)
    model = AttU_Net(ch_in, 2).to(device)


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
        # SaveImaged(
        #     keys="pred",
        #     meta_keys="pred_meta_dict",
        #     output_dir=pred_dir,
        #     output_postfix="seg",
        #     resample=False,
        #     separate_folder=False)
    ])

    if ddp:
        model.load_state_dict(ddp_state_dict(model_path))
    else:
        model.load_state_dict(torch.load(model_path))

    loader = LoadImage(image_only=True)
    loader_meta = LoadImage(image_only=False)

    model.eval()

    results = pd.DataFrame(columns=['id',
                                    'dice',
                                    'size',
                                    'size_pred',
                                    'px_x',
                                    'px_y',
                                    'px_z',
                                    'size_ml',
                                    'size_pred_ml'])
    results['id'] = ['test_' + str(item).zfill(3) for item in range(len(test_loader))]

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs= test_data["image"].to(device)

            test_data["pred"] = model(test_inputs)

            prob = f.softmax(test_data["pred"], dim=1)  # probability of infarct
            test_data["proba"] = prob

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

            test_output, test_label, test_image, test_proba = from_engine(
                ["pred", "label", "image", "proba"])(test_data)

            a = dice_metric(y_pred=test_output, y=test_label)
            dice_score = round(a.item(), 4)
            print(f"Dice score for image: {dice_score:.4f}")

            original_image = loader_meta(test_data[0]["image_meta_dict"]["filename_or_obj"])
            volx, voly, volz = original_image[1]['pixdim'][1:4]  # meta data
            pixel_vol = volx * voly * volz

            ground_truth = test_label[0][1].detach().numpy()
            prediction = (test_proba[0][1].detach().numpy() >= 0.5 ) *1
            prediction_70 = (test_proba[0][1].detach().numpy() >=0.7) *1
            prediction_90 =(test_proba[0][1].detach().numpy() >=0.9) *1

            size = ground_truth.sum()
            size_ml = size * pixel_vol / 1000

            size_pred = prediction.sum()
            size_pred_ml = size_pred * pixel_vol / 1000


            name = "test_" + os.path.basename(
                test_data[0]["image_meta_dict"]["filename_or_obj"]).split('.nii.gz')[0].split('_')[1]
            subject = ctp_df.loc[[name], "subject"].values[0]
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
    results_join.to_csv(directory + 'out_' + out_tag + '/results.csv', index=False)

if __name__ == '__main__':
    HOMEDIR = os.path.expanduser("~/")
    if os.path.exists(HOMEDIR + 'mediaflux/'):
        directory = HOMEDIR + 'mediaflux/data_freda/ctp_project/CTP_DL_Data/'
        ctp_df = pd.read_csv(
            HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
            usecols=['subject', 'segmentation_type', 'dl_id'],
        index_col='dl_id')
    elif os.path.exists('/media/mbcneuro'):
        directory = '/media/mbcneuro/CTP_DL_Data/'
        ctp_df = pd.read_csv(
            HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
            usecols=['subject', 'segmentation_type', 'dl_id'],
        index_col='dl_id')
    elif os.path.exists('Z:/data_freda'):
        directory = 'Z:/data_freda/ctp_project/CTP_DL_Data/'
        ctp_df = pd.read_csv(
            'C:/Users/fwerdiger/PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
            usecols=['subject', 'segmentation_type', 'dl_id'],
        index_col='dl_id')
    elif os.path.exists('D:'):
        directory = 'D:/ctp_project_data/CTP_DL_Data/'
        ctp_df = pd.read_csv(
            'C:/Users/fwerdiger/PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
            usecols=['subject', 'segmentation_type', 'dl_id'],
        index_col='dl_id')
    elif os.path.exists('/data/gpfs/projects/punim1086/'):
        directory = '/data/gpfs/projects/punim1086/ctp_project/CTP_DL_Data/'
        ctp_df = pd.read_csv(
            '/data/gpfs/projects/punim1086/study_design/study_lists/data_for_ctp_dl.csv',
            usecols=['subject', 'segmentation_type', 'dl_id'],
        index_col='dl_id')

    # but all the test subjects are manual segmentations so this can be removed
    # TODO: remove reference to no seg data
    main(directory, ctp_df, ddp=False)
