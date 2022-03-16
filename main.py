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
from monai.networks.nets import SegResNet
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
    Resized
)
from monai.utils import first, set_determinism

import torch

print_config()

directory = '/media/mbcneuro/HDD1/DWI_Training_Data_INSP/'
# directory = 'D:/ctp_project_data/DWI_Training_Data_INSP/'
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
        EnsureTyped(keys=["mask", "label"]),
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
