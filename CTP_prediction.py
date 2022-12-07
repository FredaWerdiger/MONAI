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
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    SaveImaged
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
    # HOME MANY TRAINING FILES ARE MANUALLY SEGMENTED
    train_df = ctp_dl_df[ctp_dl_df.apply(lambda x: 'train' in x.dl_id, axis=1)]
    num_semi_train = len(train_df[train_df.apply(lambda x: x.segmentation_type == "semi_automated", axis=1)])

    val_df = ctp_dl_df[ctp_dl_df.apply(lambda x: 'val' in x.dl_id, axis=1)]
    num_semi_val = len(val_df[val_df.apply(lambda x: x.segmentation_type == "semi_automated", axis=1)])

    # model parameters
    max_epochs = 300
    image_size = (32, 32, 32)
    patch_size = None
    batch_size = 2
    val_interval = 2
    vis_interval = 100
    out_tag = 'unet_cam'
    if not os.path.exists(directory + 'out_' + out_tag):
        os.makedirs(directory + 'out_' + out_tag)

    set_determinism(seed=42)

    train_files = BuildDataset(directory, 'train').images_dict
    val_files = BuildDataset(directory, 'validation').images_dict
    test_files = BuildDataset(directory, 'test').no_seg_dict

    # IMAGES SHOULD NOT BE DOWNSAMPLED
    # RANDOM SAMPLE OF PATCHES BETTER
    # IMAGES ARE ORIGINALLY 512X512X320  or so

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"],
                    mode=['trilinear', "nearest"],
                    align_corners=[True, None],
                    spatial_size=image_size),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # RandCropByPosNegLabeld(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=patch_size,
            #     pos=0.75,
            #     neg=0.25,
            #     num_samples=4,
            #     image_key="image",
            #     image_threshold=0,
            # ),
            RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"],
                    mode=['trilinear', "nearest"],
                    align_corners=[True, None],
                    spatial_size=image_size),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys="image"),
            Resized(keys="image",
                    mode='trilinear',
                    align_corners=True,
                    spatial_size=image_size),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
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

    # test_ds = Dataset(data=test_files, transform=test_transforms)[:1]
    # test_loader = DataLoader(test_ds, batch_size=1)

    # Uncomment to display data
    #
    # import random
    # m = random.randint(0, len(val_files)-1)
    # s = random.randint(0, image_size[0]-1)
    # val_data_example = val_dataset[m]
    # print(f"image shape: {val_data_example['image'].shape}")
    # plt.figure("image", (30, 6))
    # for i in range(4):
    #     plt.subplot(1, 5, i + 1)
    #     plt.title(f"image channel {i}")
    #     plt.imshow(val_data_example["image"][i, :, :, s].detach().cpu(), cmap="gray")
    # # also visualize the 3 channels label corresponding to this image
    # print(f"label shape: {val_data_example['label'].shape}")
    # plt.subplot(1, 5, 5)
    # plt.title("label")
    # plt.imshow(val_data_example["label"][0, :, :, s].detach().cpu())
    # plt.show()
    # plt.close()

    device = 'cuda'
    channels = (16, 32, 64)
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=2,
        channels=channels,
        strides=(2, 2),
        num_res_units=2,
        norm=Norm.BATCH
    )

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
    # dice_metric_torch_macro = Dice(dist_sync_on_step=True,
    #                                num_classes=2, ignore_index=0, average='macro').to(device)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


    epoch_loss_values = []
    dice_metric_values = []
    best_metric = -1
    best_metric_epoch = -1

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    start = time.time()
    model_name = 'best_metric_model' + str(max_epochs) + '.pth'

    # set up visualisation
    cam = GradCAM(nn_module=model, target_layers="model.2.0.adn.A")
    visual = []
    visual_orig = []
    visual_names = []

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        epoch_loss = 0
        step = 0
        model.train()
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            # commenting out print function
            # print(
            #     f"{step}/{len(train_dataset) // train_loader.batch_size}, "
            #     f"train_loss: {loss.item():.4f}")
            if (epoch + 1) % vis_interval == 0 and step == 1:
                cam_results = cam(x=inputs).cpu().numpy()
                visual.append(cam_results[0][0][:, :, int(np.ceil(image_size[1]/2))])
                visual_orig.append(batch_data["image"][0][1][:, :, int(np.ceil(image_size[1]/2))])
                name = "train_" + os.path.basename(batch_data["image_meta_dict"]["filename_or_obj"][0]).split('.nii.gz')[0].split('_')[1]
                subject = ctp_dl_df.loc[ctp_dl_df.dl_id == name, "subject"].values[0]
                visual_names.append(subject)
        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            print("Evaluating...")
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    # unsure how to optimize this
                    roi_size = image_size
                    sw_batch_size = 1
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)

                    # compute metric for current iteration
                    # dice_metric_torch_macro(val_outputs, val_labels.long())
                    # now to for the MONAI dice metric
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(val_outputs, val_labels)

                mean_dice = dice_metric.aggregate().item()
                dice_metric.reset()
                dice_metric_values.append(mean_dice)

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
                # # evaluate during training process
                # model.load_state_dict(torch.load(
                #     os.path.join(directory, 'out_' + out_tag, model_name)))
                # model.eval()
                # with torch.no_grad():
                #     for i, val_data in enumerate(val_loader):
                #         roi_size = image_size
                #         sw_batch_size = 1
                #         val_outputs = sliding_window_inference(
                #             val_data["image"].to(device), roi_size, sw_batch_size, model
                #         )
                #         # plot the slice [:, :, 80]
                #         plt.figure("check", (30, 6))
                #         plt.subplot(1, 3, 1)
                #         plt.title(f"image {i}")
                #         plt.imshow(val_data["image"][0, 0, :, :, int(image_size[0]/2)], cmap="gray")
                #         plt.subplot(1, 3, 2)
                #         plt.title(f"label {i}")
                #         plt.imshow(val_data["label"][0, 0, :, :, int(image_size[0]/2)])
                #         plt.subplot(1, 3, 3)
                #         plt.title(f"output {i}")
                #         plt.imshow(torch.argmax(
                #             val_outputs, dim=1).detach().cpu()[0, :, :, int(image_size[0]/2)])
                #         plt.show()
                #         if i == 2:
                #             break
        del loss, outputs     # get rid of memory
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
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(dice_metric_values))]
    y = dice_metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, 'b', label="Dice")
    plt.savefig(os.path.join(directory + 'out_' + out_tag, model_name.split('.')[0] + 'plot_loss.png'),
                bbox_inches='tight', dpi=300, format='png')
    plt.close()

    fig, ax = plt.subplots(2, len(visual), figsize=(len(visual), 2))
    for i, vis_name in enumerate(zip(visual, visual_names)):
        vis, name = vis_name
        ax[1, i].imshow(visual_orig[i], cmap='gray')
        ax[1, i].axis('off')
        im_1 = ax[0, i].imshow(vis, cmap='jet')
        ax[0, i].axis('off')
        ax[0, i].set_title(f"{name}: epoch {(i * vis_interval) + vis_interval}", fontsize='4')
        fig.colorbar(im_1, ax=ax.ravel, shrink=0.25)
    plt.savefig(os.path.join(directory + 'out_' + out_tag, model_name.split('.')[0] + 'visuals.png'),
                bbox_inches='tight', dpi=300, format='png')
    plt.close()

    # # TESTING
    #
    # # LOCATION TO SAVE OUTPUT
    # prob_dir = os.path.join(directory + 'out_' + out_tag, "proba_masks")
    # if not os.path.exists(prob_dir):
    #     os.makedirs(prob_dir)
    #
    # test_transforms = Compose(
    #     [
    #         LoadImaged(keys=["image"]),
    #         EnsureChannelFirstd(keys="image"),
    #         Resized(keys="image",
    #                 mode='trilinear',
    #                 align_corners=True,
    #                 spatial_size=image_size),
    #         NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    #         EnsureTyped(keys=["image"]),
    #     ]
    # )
    #
    # post_transforms = Compose([
    #     EnsureTyped(keys=["pred"]),
    #     Invertd(
    #         keys=["pred", "proba"],
    #         transform=test_transforms,
    #         orig_keys=["image", "image"],
    #         meta_keys=["pred_meta_dict", "pred_meta_dict"],
    #         orig_meta_keys=["image_meta_dict", "image_meta_dict"],
    #         meta_key_postfix="meta_dict",
    #         nearest_interp=[False, False],
    #         to_tensor=[True, True],
    #     ),
    #     SaveImaged(
    #         keys="proba",
    #         meta_keys="pred_meta_dict",
    #         output_dir=prob_dir,
    #         output_postfix="proba",
    #         resample=False,
    #         separate_folder=False)
    # ])
    # test_ds = Dataset(data=test_files, transform=test_transforms)[:1]
    # test_loader = DataLoader(test_ds, batch_size=1)

    # # LOAD THE BEST MODEL
    # model.load_state_dict(torch.load(os.path.join(
    #     directory, 'out_' + out_tag, model_name)))
    # model.eval()
    #
    # with torch.no_grad():
    #     for i, test_data in enumerate(test_loader):
    #         test_inputs = test_data["image"].to(device)
    #         roi_size = (64, 64, 64)
    #         sw_batch_size = 2
    #         test_data["pred"] = sliding_window_inference(
    #             test_inputs, roi_size, sw_batch_size, model)
    #         prob = f.softmax(test_data["pred"], dim=1) # probability of infarct
    #         test_data["proba"] = prob
    #         test_data = [post_transforms(i) for i in decollate_batch(test_data)]
    #         # test_pred, test_proba= from_engine(["pred", "proba"])(test_data)
    #         #
    #         # original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])

if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    main()
