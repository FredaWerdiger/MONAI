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
from monai.utils import first, set_determinism
from torchmetrics import Dice
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
)

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import time
from numba import cuda
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from torchmetrics import Dice


def example(rank, world_size):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    HOMEDIR = os.path.expanduser('~/')
    if os.path.exists('/media/mbcneuro'):
        directory = '/media/mbcneuro/CTP_DL_Data/'
    elif os.path.exists('/media/fwerdiger'):
        directory = '/media/fwerdiger/Storage/CTP_DL_Data/'

    # model parameters
    max_epochs = 600
    image_size = (128, 128, 128)
    batch_size = 2
    val_interval = 2
    out_tag = 'unet_manual_segmentation_patients'
    if not os.path.exists(directory + 'out_' + out_tag):
        os.makedirs(directory + 'out_' + out_tag)

    set_determinism(seed=42)

    train_files = BuildDataset(directory, 'train').images_dict
    val_files = BuildDataset(directory, 'validation').images_dict

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"],
                    mode=['trilinear', "nearest"],
                    align_corners=[True, None],
                    spatial_size=image_size),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
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

    train_dataset = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=4
    )

    val_dataset = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=4
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

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


    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=2,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH
    ).to(rank)

    model = DDP(model, device_ids=[rank])

    loss_function = DiceLoss(smooth_dr=1e-5,
                             smooth_nr=0,
                             to_onehot_y=True,
                             softmax=True,
                             include_background=False)

    optimizer = Adam(model.parameters(),
                     1e-4,
                     weight_decay=1e-5)

    dice_metric = DiceMetric(include_background=False, reduction='mean')
    dice_metric_torch_macro = Dice(dist_sync_on_step=True,
                                   num_classes=2, ignore_index=0, average='macro').to(rank)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    if rank == 0:
        epoch_loss_values = []
        dice_metric_values = []
        dice_metric_macro_values = []
        best_metric = -1
        best_metric_epoch = -1
        best_metric_macro = -1
        best_metric_epoch_macro = -1

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
            inputs, labels = (
                batch_data["image"].to(rank),
                batch_data["label"].to(rank),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            # commenting out print function
            # print(
            #     f"{step}/{len(train_ds) // train_loader.batch_size}, "
            #     f"train_loss: {loss.item():.4f}")
        lr_scheduler.step()
        if rank == 0:
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            print("Evaluating...")
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(rank),
                        val_data["label"].to(rank),
                    )
                    # unsure how to optimize this
                    roi_size = image_size
                    sw_batch_size = 1
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)

                    # compute metric for current iteration
                    dice_metric_torch_macro(val_outputs, val_labels.long())
                    # now to for the MONAI dice metric
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(val_outputs, val_labels)

                torchmetric_macro = dice_metric_torch_macro.compute().item()
                dice_metric_torch_macro.reset()
                mean_dice = dice_metric.aggregate().item()
                dice_metric.reset()

                if rank == 0:
                    dice_metric_values.append(mean_dice)
                    dice_metric_macro_values.append(torchmetric_macro)

                    if mean_dice > best_metric:
                        best_metric = mean_dice
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), os.path.join(
                            directory, 'out_' + out_tag, model_name))
                        print("saved new best metric model")
                    if torchmetric_macro > best_metric_macro:
                        best_metric_macro = torchmetric_macro
                        best_metric_epoch_macro = epoch + 1

                    print(
                        f"current epoch: {epoch + 1} current mean dice: {mean_dice:.4f}"
                        f"\ncurrent torchmetric macro: {torchmetric_macro:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                        f"\nbest mean dice macro: {best_metric_macro:.4f} "
                        f"at epoch: {best_metric_epoch_macro}"
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
                #             val_data["image"].to(rank), roi_size, sw_batch_size, model
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
                #
    end = time.time()
    time_taken = end - start
    print(f"Time taken: {round(time_taken, 0)} seconds")
    time_taken_hours = time_taken/3600
    time_taken_mins = np.ceil((time_taken/3600 - int(time_taken/3600)) * 60)
    time_taken_hours = int(time_taken_hours)

    with open(directory + 'out_' + out_tag + '/model_info.txt', 'w') as myfile:
        myfile.write(f'Train dataset size: {len(train_dataset)}\n')
        myfile.write(f'validation dataset size: {len(val_dataset)}\n')
        myfile.write(f'Number of epochs: {max_epochs}\n')
        myfile.write(f'Batch size: {batch_size}\n')
        myfile.write(f'Image size: {image_size}\n')
        myfile.write(f'Validation interval: {val_interval}\n')
        myfile.write(f"Best metric: {best_metric:.4f}\n")
        myfile.write(f"Best metric epoch: {best_metric_epoch}\n")
        myfile.write(f"Best metric other: {best_metric:.4f}\n")
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
    plt.plot(x, y, 'b', label="MONAI Dice")
    y = dice_metric_macro_values
    plt.plot(x, y, 'r', label="torchmetrics macro mean dice")
    plt.legend(loc=4)
    plt.savefig(os.path.join(directory + 'out_' + out_tag, model_name.split('.')[0] + 'plot_loss.png'),
                bbox_inches='tight', dpi=300, format='png')
    plt.close()
    cleanup()


def main():
    # comment out below for dev
    free_gpu_cache()
    world_size = 2
    mp.spawn(example,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    main()
