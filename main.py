# following tutorial from BRATs segmentation
# two classes insead of 4 classes
import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
from monai.config import print_config
from monai.data import Dataset, CacheDataset, DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.apps import download_and_extract
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet, UNet
from monai.networks.layers import Norm
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    ScaleIntensityRanged,
    MapTransform,
    NormalizeIntensityd,
    ScaleIntensityRangePercentilesd,
    Orientationd,
    RandFlipd,
    AddChanneld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    Resized,
    SaveImaged
)
from monai.utils import first, set_determinism

import torch

from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

free_gpu_cache()

print_config()

directory = '/media/mbcneuro/HDD1/DWI_Training_Data_INSP/'
#directory = 'D:/ctp_project_data/DWI_Training_Data_INSP/'
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


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


train_files, val_files = [make_dict(root_dir, string) for string in ['train', 'validation']]

set_determinism(seed=42)

train_transforms = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Resized(keys=["image", "label"],
                mode=['trilinear', "nearest"],
                align_corners=[True, None],
                spatial_size=(128, 128, 128)),
        ScaleIntensityRangePercentilesd(keys="image",
                                        lower=1,
                                        upper=99,
                                        b_min=0.0,
                                        b_max=1.0,
                                        channel_wise=True,
                                        clip=True),
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
                spatial_size=(128, 128, 128)),
        ScaleIntensityRangePercentilesd(keys="image",
                                        lower=1,
                                        upper=99,
                                        b_min=0.0,
                                        b_max=1.0,
                                        channel_wise=True,
                                        clip=True),
        EnsureTyped(keys=["image", "label"]),
    ]
)

# here we don't cache any data in case out of memory issue
train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=1.0,
    num_workers=4
)


train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=4)

val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_rate=1.0,
    num_workers=4)

val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4)

import random
m = random.randint(0, 45)
s = random.randint(20, 100)
val_data_example = val_ds[m]
print(f"image shape: {val_data_example['image'].shape}")
plt.figure("image", (18, 6))
for i in range(2):
    plt.subplot(1, 3, i + 1)
    plt.title(f"image channel {i}")
    plt.imshow(val_data_example["image"][i, :, :, s].detach().cpu(), cmap="gray")
# also visualize the 3 channels label corresponding to this image
print(f"label shape: {val_data_example['label'].shape}")
plt.subplot(1, 3, 3)
plt.title("label")
plt.imshow(val_data_example["label"][0, :, :, s].detach().cpu())
plt.show()
plt.close()

device = torch.device("cuda:0")
# model = UNet(in_channels=2,
#              out_channels=2,
#              n_blocks=4,
#              start_filters=32,
#              activation='relu',
#              normalization='batch',
#              conv_mode='same',
#              dim=3).to(device)
# Building a model now

model = UNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=2,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

max_epochs = 300
val_interval = 4
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
start = time.time()
model_name = 'best_metric_model' + str(max_epochs) + '.pth'
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
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
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    root_dir, model_name))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
end = time.time()
time_taken = end - start
print(f"Time taken: {round(time_taken, 0)} seconds")
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.show()
plt.savefig(os.path.join(root_dir, model_name.split('.')[0] + 'plot_loss.png'))

model.load_state_dict(torch.load(
    os.path.join(root_dir, model_name)))
model.eval()

with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        i = str(i).zfill(3)

        roi_size = (160, 160, 160)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
            val_data["image"].to(device), roi_size, sw_batch_size, model
        )
        # plot the slice [:, :, 80]
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {i}")
        plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {i}")
        plt.imshow(val_data["label"][0, 0, :, :, 80])
        plt.subplot(1, 3, 3)
        plt.title(f"output {i}")
        plt.imshow(torch.argmax(
            val_outputs, dim=1).detach().cpu()[0, :, :, 80])
        plt.show()
        if i ==10:
            break

test_images = sorted(
    glob.glob(os.path.join(root_dir, "test", "images", "*.nii.gz")))

test_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Resized(keys="image",
                mode='trilinear',
                align_corners= True,
                spatial_size=(128, 128, 128)),
        ScaleIntensityRangePercentilesd(keys="image",
                                        lower=1,
                                        upper=99,
                                        b_min=0.0,
                                        b_max=1.0,
                                        channel_wise=True,
                                        clip=True),
        EnsureTyped(keys="image"),
    ]
)

test_data = [{"image": image} for image in test_images]
test_ds = Dataset(
    data=test_data, transform=test_transforms)

test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

post_transforms = Compose([
    EnsureTyped(keys="pred"),
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
    AsDiscreted(keys="pred", argmax=True),
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=root_dir + "out", output_postfix="seg", resample=False),
])
# uncomment the following lines to visualize the predicted results
from monai.transforms import LoadImage
loader = LoadImage()

model.load_state_dict(torch.load(
    os.path.join(root_dir, model_name)))

model.eval()

with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        test_inputs = test_data["image"].to(device)
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        test_data["pred"] = sliding_window_inference(
            test_inputs, roi_size, sw_batch_size, model)

        test_data = [post_transforms(i) for i in decollate_batch(test_data)]

        # uncomment the following lines to visualize the predicted results
        test_output = from_engine(["pred"])(test_data)

        original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])[0]

        plt.figure("check", (18, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image[:, :, 10, 0], cmap="gray")
        plt.title(f"image {i}")
        plt.subplot(1, 2, 2)
        plt.imshow(test_output[0].detach().cpu()[0, :, :, 10])
        plt.title(f"prediction {i}")
        plt.show()