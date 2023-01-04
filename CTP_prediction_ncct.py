import os
from monai_fns import *
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.handlers import EarlyStopHandler
from monai.handlers.utils import from_engine
from monai.utils import first, set_determinism
# from torchmetrics import Dice
from monai.visualize import GradCAM
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
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
# from numba import cuda
import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as f
from torch.optim import Adam
from recursive_data import get_semi_dataset
from models import U_Net, U_NetCT
from monai.utils import (
    BlendMode,
    PytorchPadMode
)


def main():
    HOMEDIR = os.path.expanduser('~/')
    if os.path.exists(HOMEDIR + 'mediaflux/'):
        directory = HOMEDIR + 'mediaflux/data_freda/ctp_project/CTP_DL_Data/'
        ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])
    elif os.path.exists('/data/gpfs/projects/punim1086/ctp_project'):
        directory = '/data/gpfs/projects/punim1086/ctp_project/CTP_DL_Data/'
        ctp_dl_df = pd.read_csv('/data/gpfs/projects/punim1086/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])
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
    # HOME MANY TRAINING FILES ARE MANUALLY SEGMENTED
    train_df = ctp_dl_df[ctp_dl_df.apply(lambda x: 'train' in x.dl_id, axis=1)]
    num_semi_train = len(train_df[train_df.apply(lambda x: x.segmentation_type == "semi_automated", axis=1)])

    val_df = ctp_dl_df[ctp_dl_df.apply(lambda x: 'val' in x.dl_id, axis=1)]
    num_semi_val = len(val_df[val_df.apply(lambda x: x.segmentation_type == "semi_automated", axis=1)])

    # model parameters
    max_epochs = 400
    image_size = (128, 128, 128)
    patch_size = None
    batch_size = 2
    val_interval = 2
    atrophy = True
    out_tag = 'unet_simple'
    out_tag = out_tag + '_atrophy' if atrophy else out_tag + '_raw_ncct'
    if not os.path.exists(directory + 'out_' + out_tag):
        os.makedirs(directory + 'out_' + out_tag)

    set_determinism(seed=42)

    train_files = BuildDataset(directory, 'train').ncct_dict
    val_files = BuildDataset(directory, 'validation').ncct_dict

    transform_dir = os.path.join(directory, 'out_' + out_tag, 'ncct_trans')
    if not os.path.exists(transform_dir):
        os.makedirs(transform_dir)
    if atrophy:
        atrophy_transforms = [
            ThresholdIntensityd(keys="ncct", threshold=40, above=False),
            ThresholdIntensityd(keys="ncct", threshold=0, above=True),
            GaussianSmoothd(keys="ncct", sigma=1)]
            # SaveImaged(keys="ncct",
            #            output_dir=transform_dir,
            #            meta_keys="ncct_meta_dict",
            #            output_postfix="transform",
            #            resample=False,
            #            separate_folder=False)]
    else:
        atrophy_transforms = []
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "ncct", "label"]),
            EnsureChannelFirstd(keys=["image", "ncct", "label"]),
            Resized(keys=["image", "ncct", "label"],
                    mode=['trilinear', 'trilinear', "nearest"],
                    align_corners=[True, True, None],
                    spatial_size=image_size),
            *atrophy_transforms,
            NormalizeIntensityd(keys=["image", "ncct"], nonzero=True, channel_wise=True),
            RandAffined(keys=['image', "ncct", 'label'], prob=0.5, translate_range=10),
            RandFlipd(keys=["image", "ncct", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "ncct", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "ncct", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=["image", "ncct"], factors=0.1, prob=1.0),
            RandShiftIntensityd(keys=["image", "ncct"], offsets=0.1, prob=1.0),
            EnsureTyped(keys=["image", "ncct", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "ncct", "label"]),
            EnsureChannelFirstd(keys=["image", "ncct", "label"]),
            Resized(keys=["image", "ncct", "label"],
                    mode=['trilinear', 'trilinear', "nearest"],
                    align_corners=[True, True, None],
                    spatial_size=image_size),
            *atrophy_transforms,
            NormalizeIntensityd(keys=["image", "ncct"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "ncct", "label"]),
        ]
    )

    train_dataset = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=8)

    val_dataset = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=8)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            pin_memory=True)

    # # sanity check to see everything is there
    # s = 50
    # data_example = val_dataset[1]
    # print(f"image shape: {data_example['image'].shape}")
    # plt.figure("image", (18, 4))
    # for i in range(4):
    #     plt.subplot(1, 6, i + 1)
    #     plt.title(f"image channel {i}")
    #     plt.imshow(data_example["image"][i, :, :, s].detach().cpu(), cmap="jet")
    #     plt.axis('off')
    # plt.subplot(1, 6, 5)
    # plt.imshow(data_example["ncct"][0,:, :, s].detach().cpu(), cmap="jet")
    # plt.title("ncct")
    # plt.axis('off')
    # print(f"label shape: {data_example['label'].shape}")
    # plt.subplot(1, 6, 6)
    # plt.title("label")
    # plt.axis('off')
    # plt.imshow(data_example["label"][0, :, :, s].detach().cpu(), cmap="jet")
    # plt.show()
    # plt.close()

    device = 'cuda'
    channels = (16, 32, 64)

    model = U_NetCT(img_ch=4,output_ch=2)

    # model = torch.nn.DataParallel(model)
    model.to(device)

    loss_function = DiceLoss(smooth_dr=1e-5,
                             smooth_nr=0,
                             to_onehot_y=True,
                             softmax=True,
                             include_background=False)

    optimizer = Adam(model.parameters(),
                     1e-4,
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
    model_name = 'best_metric_model' + str(max_epochs) + '.pth'

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        epoch_loss = 0
        step = 0
        model.train()
        for batch_data in train_loader:
            step += 1
            inputs, nccts, labels = (
                batch_data["image"].to(device),
                batch_data["ncct"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs, nccts)
            loss = loss_function(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            print("Evaluating...")
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_nccts, val_labels = (
                        val_data["image"].to(device),
                        val_data["ncct"].to(device),
                        val_data["label"].to(device),
                    )
                    # unsure how to optimize this
                    roi_size = (128, 128, 128)
                    sw_batch_size = batch_size
                    args = [val_nccts]
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model,
                        0.25,
                        BlendMode.CONSTANT,
                        0.125,
                        PytorchPadMode.CONSTANT,
                        0.0,
                        None,
                        None,
                        False,
                        None,
                        *args)

                    # compute metric for current iteration
                    # dice_metric_torch_macro(val_outputs, val_labels.long())
                    # now to for the MONAI dice metric
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(val_outputs, val_labels)
                for val_data in train_loader:
                    val_inputs, val_nccts, val_labels = (
                        val_data["image"].to(device),
                        val_data["ncct"].to(device),
                        val_data["label"].to(device),
                    )
                    # unsure how to optimize this
                    roi_size = (128, 128, 128)
                    sw_batch_size = batch_size
                    args = [val_nccts]
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model,
                        0.25,
                        BlendMode.CONSTANT,
                        0.125,
                        PytorchPadMode.CONSTANT,
                        0.0,
                        None,
                        None,
                        False,
                        None,
                        *args)

                    # compute metric for current iteration
                    # dice_metric_torch_macro(val_outputs, val_labels.long())
                    # now to for the MONAI dice metric
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric_train(val_outputs, val_labels)

                mean_dice = dice_metric.aggregate().item()
                dice_metric.reset()
                dice_metric_values.append(mean_dice)
                mean_dice_train = dice_metric_train.aggregate().item()
                dice_metric_train.reset()
                dice_metric_values_train.append(mean_dice_train)

                if mean_dice > best_metric:
                    best_metric = mean_dice
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        directory, 'out_' + out_tag, model_name))
                    print("saved new best metric model")

                print(
                    f"current epoch: {epoch + 1} current mean dice: {mean_dice:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
        del loss, outputs
    end = time.time()
    time_taken = end - start
    print(f"Time taken: {round(time_taken, 0)} seconds")
    time_taken_hours = time_taken/3600
    time_taken_mins = np.ceil((time_taken/3600 - int(time_taken/3600)) * 60)
    time_taken_hours = int(time_taken_hours)

    with open(directory + 'out_' + out_tag + '/model_info_' + str(max_epochs) + '.txt', 'w') as myfile:
        myfile.write(f'Train dataset size: {len(train_files)}\n')
        myfile.write(f'Train semi-auto segmented: {num_semi_train}\n')
        myfile.write(f'Validation dataset size: {len(val_files)}\n')
        myfile.write(f'Validation semi-auto segmented: {num_semi_val}\n')
        myfile.write("Atrophy filter used? ")
        if atrophy:
            myfile.write("yes")
        else:
            myfile.write("no")
        myfile.write(f'Number of epochs: {max_epochs}\n')
        myfile.write(f'Batch size: {batch_size}\n')
        myfile.write(f'Image size: {image_size}\n')
        myfile.write(f'Patch size: {patch_size}\n')
        myfile.write(f'channels: {channels}\n')
        myfile.write(f'Validation interval: {val_interval}\n')
        myfile.write(f"Best metric: {best_metric:.4f}\n")
        myfile.write(f"Best metric epoch: {best_metric_epoch}\n")
        myfile.write(f"Time taken: {time_taken_hours} hours, {time_taken_mins} mins\n")

    # plot things
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Average Loss per Epoch")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Mean Dice (Accuracy)")
    x = [val_interval * (i + 1) for i in range(len(dice_metric_values))]
    y = dice_metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, 'b', label="Dice on validation data")
    y = dice_metric_values_train
    plt.plot(x, y, 'k', label="Dice of train data")
    plt.legend(loc="center right")
    plt.savefig(os.path.join(directory + 'out_' + out_tag, model_name.split('.')[0] + 'plot_loss.png'),
                bbox_inches='tight', dpi=300, format='png')
    plt.close()


if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    main()

