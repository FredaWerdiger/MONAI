# following tutorial from BRATs segmentation
# two classes insead of 4 classes
import os
import pandas as pd
import sys
sys.path.append('/data/gpfs/projects/punim1086/ctp_project/MONAI/')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import math
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.metrics import f1_score
from monai.config import print_config
from monai.data import Dataset, CacheDataset, DataLoader, decollate_batch
from monai.handlers import EarlyStopHandler
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet, UNet, AttentionUnet, DenseNet
from monai.networks.layers import Norm
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImage,
    LoadImaged,
    CropForegroundd,
    NormalizeIntensityd,
    ScaleIntensityRangePercentilesd,
    RandAffined,
    RandFlipd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    Resized,
    SaveImaged,
    SplitDimd
)

from monai.utils import first, set_determinism
from monai_fns import *
from densenet import DenseNetFCN
import ignite
import torch
import os
from recursive_data import get_semi_dataset
from torch.nn import DataParallel as DDP




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



def main():

    directory = '/data/gpfs/projects/punim1086/ctp_project/DWI_Training_Data/'
    existing_model = directory + 'out_unet_recursive/best_metric_model600.pth'

    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)

    # create outdir
    out_tag = "densenetFCN"
    if not os.path.exists(root_dir + 'out_' + out_tag):
        os.makedirs(root_dir + 'out_' + out_tag)

    train_files, val_files, test_files = [
        make_dict(root_dir, string) for string in ['train', 'validation', 'test']]
    semi_files = get_semi_dataset()
    train_files = semi_files + train_files

    set_determinism(seed=42)

    max_epochs = 600
    batch_size = 2
    image_size = (128, 128, 128)
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
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=1.0),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
            # RandAdjustContrastd(keys="image", prob=1, gamma=(0.5, 1)),
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

    # here we don't cache any data in case out of memory issue
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=0
    )


    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=0)

    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            pin_memory=True)

    rank = 'cuda'

    # Uncomment to display data

    import random
    m = random.randint(0, 50)
    s = random.randint(0, 63)
    val_data_example = val_ds[m]
    # print(f"image shape: {val_data_example['image_b1000'].shape}")
    # plt.figure("image", (18, 6))
    # for i in range(1):
    #     plt.subplot(1, 3, i + 1)
    #     plt.title(f"image channel {i}")
    #     plt.imshow(val_data_example["image_b1000"][i, :, :, s].detach().cpu(), cmap="gray")
    # # also visualize the 3 channels label corresponding to this image
    # print(f"label shape: {val_data_example['label'].shape}")
    # plt.subplot(1, 3, 3)
    # plt.title("label")
    # plt.imshow(val_data_example["label"][0, :, :, s].detach().cpu())
    # plt.show()
    # plt.close()

    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(rank)
    model = DenseNetFCN(
        ch_in=2,
        ch_out_init=48,
        num_classes=2,
        growth_rate=16,
        layers=(4, 5, 7, 10, 12),
        bottleneck=True,
        bottleneck_layer=15
    )
    # model = AttentionUnet(
    #     spatial_dims=3,
    #     in_channels=2,
    #     out_channels=2,
    #     channels=(32, 64, 128, 256, 512),
    #     strides=(2, 2, 2, 2),
    # ).to(rank)

    # model = DenseNet(
    #     spatial_dims=3,
    #     in_channels=2,
    #     out_channels=2
    # ).to(rank)

    # model = SegResNet(
    #     blocks_down=[1, 2, 2, 4],
    #     blocks_up=[1, 1, 1],
    #     init_filters=8,
    #     in_channels=2,
    #     out_channels=2,
    #     dropout_prob=0.2,
    # ).to(device)

    # model = DDP(model)
    model = model.to(rank)

    loss_function = DiceLoss(
        smooth_nr=0,
        smooth_dr=1e-5,
        to_onehot_y=True,
        softmax=True,
        include_background=False)
    optimizer = torch.optim.Adam(
        model.parameters(),
        1e-3,
        weight_decay=1e-4)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    val_interval = 2
    # only doing these for master node
    epoch_loss_values = []
    metric_values = []
    best_metric = -1
    best_metric_epoch = -1
    # Below not needed for torchmetrics metric
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    start = time.time()
    model_name = 'best_metric_model' + str(max_epochs) + '.pth'
    # load existing model
    # model.load_state_dict(torch.load(existing_model))

    trainer = ignite.engine.create_supervised_trainer(model, optimizer, loss_function, rank, False)
    handler = EarlyStopHandler(
        trainer=trainer,
        patience=1,
        score_function=lambda x: x.state.metrics['val_mean_dice'],
    )

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
        # lr_scheduler.step()
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
                    val_outputs = model(val_inputs)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(val_outputs, val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()

                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        root_dir, 'out_' + out_tag, model_name))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

    end = time.time()
    time_taken = end - start
    print(f"Time taken: {round(time_taken, 0)} seconds")
    time_taken_hours = time_taken/3600
    time_taken_mins = np.ceil((time_taken/3600 - int(time_taken/3600)) * 60)
    time_taken_hours = int(time_taken_hours)

    model_name = model._get_name()
    loss_name = loss_function._get_name()
    # generate loss plot
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    loss_values_df = pd.DataFrame(columns=["epoch", "average_loss"])
    loss_values_df['average_loss'] = y
    loss_values_df['epoch'] = x
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    dice_values_df = pd.DataFrame(columns=["epoch", "average_loss"])
    dice_values_df['average_loss'] = y
    dice_values_df['epoch'] = x
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig(os.path.join(root_dir + 'out_' + out_tag, str(max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_plot_loss.png'),
                bbox_inches='tight', dpi=300, format='png')
    plt.close()

    # save model results in a separate file
    with open(root_dir + 'out_' + out_tag + '/model_info_' + str(max_epochs) + '_epoch_' + model_name + '_' + loss_name + '.txt', 'w') as myfile:
        myfile.write(f'Train dataset size: {len(train_files)}\n')
        myfile.write(f'Semi-automated segmentations: {len(semi_files)}\n')
        myfile.write(f'Validation dataset size: {len(val_files)}\n')
        myfile.write(f'Model: {model_name}\n')
        myfile.write(f'Loss function: {loss_name}\n')
        myfile.write(f'Number of epochs: {max_epochs}\n')
        myfile.write(f'Batch size: {batch_size}\n')
        myfile.write(f'Image size: {image_size}\n')
        myfile.write(f'Validation interval: {val_interval}\n')
        myfile.write(f"Best metric: {best_metric:.4f}\n")
        myfile.write(f"Best metric epoch: {best_metric_epoch}\n")
        myfile.write(f"Time taken: {time_taken_hours} hours, {time_taken_mins} mins\n")

    dice_values_df.to_csv(root_dir + 'out_' + out_tag + '/dice_values_' + str(
        max_epochs) + '_epoch_' + model_name + '_' + loss_name + '.csv',
                          index=False)
    loss_values_df.to_csv(root_dir + 'out_' + out_tag + '/loss_values_' + str(
        max_epochs) + '_epoch_' + model_name + '_' + loss_name + '.csv',
                          index=False)

if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    main()
