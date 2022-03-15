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
from monai.data import Dataset, DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
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
    MapTransform,
    NormalizeIntensityd,
    ScaleIntensityRangePercentilesd,
    Orientationd,
    RandFlipd,
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
directory = 'D:/ctp_project_data/DWI_Training_Data_INSP/'
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
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRangePercentilesd(keys=["image"],
                                        lower=1,
                                        upper=99,
                                        b_min=0,
                                        b_max=1,
                                        channel_wise=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Resized(keys=["image", "label"],
                mode=('trilinear', 'nearest'),
                align_corners=(True, None),
                spatial_size=[2, 128, 128, 128]),
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
        LoadImaged(keys=["image", "label"],
                   dtype=np.float32),
        EnsureChannelFirstd(keys="image"),
        # ScaleIntensityRangePercentilesd(keys=["image"],
        #                                 lower=1,
        #                                 upper=99,
        #                                 b_min=0,
        #                                 b_max=1,
        #                                 channel_wise=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

        Resized(keys=["image", "label"],
                mode=('trilinear', 'nearest'),
                align_corners=(True, None),
                spatial_size=[2, 128, 128, 128]),
        EnsureTyped(keys=["image", "label"]),
    ]
)

# here we don't cache any data in case out of memory issue
train_ds = Dataset(
    data=train_files,
    transform=train_transforms
)


train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

val_ds = Dataset(
    root_dir=root_dir,
    transform=val_transform,
    section="validation",
    download=False,
    cache_rate=0.0,
    num_workers=4,
)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
