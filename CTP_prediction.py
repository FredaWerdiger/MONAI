import os
from monai_fns import DDPSetUp, BuildDataset, prepare
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
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
from numba import cuda
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam

print_config()

# model parameters
image_size = (128, 128, 128)
batch_size = 2
val_interval = 2
HOMEDIR = os.path.expanduser('~/')
directory = '/media/fwerdiger/Storage/CTP_DL_Data/'
os.path.exists(directory)

train_files = BuildDataset(directory, 'train').images_dict
val_files = BuildDataset(directory, 'validation').images_dict

rank = 'cuda'

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
    num_workers=0
)

val_dataset = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_rate=1.0,
    num_workers=0
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

# Uncomment to display data
#
import random
m = random.randint(0, len(val_files)-1)
s = random.randint(0, 127)
val_data_example = val_dataset[m]
print(f"image shape: {val_data_example['image'].shape}")
plt.figure("image", (36, 6))
for i in range(4):
    plt.subplot(1, 5, i + 1)
    plt.title(f"image channel {i}")
    plt.imshow(val_data_example["image"][i, :, :, s].detach().cpu(), cmap="gray")
# also visualize the 3 channels label corresponding to this image
print(f"label shape: {val_data_example['label'].shape}")
plt.subplot(1, 5, 5)
plt.title("label")
plt.imshow(val_data_example["label"][0, :, :, s].detach().cpu())
plt.show()
plt.close()


model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=2,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
    num_res_units=2,
    norm=Norm.Batch
).to(rank)

loss_function = DiceLoss(smooth_dr=1e-5,
                         smooth_nr=0,
                         to_onehot_y=True,
                         softmax=True,
                         include_background=False)

optimizer = Adam(model.parameters(),
                 1e-4,
                 weight_decay=1e-5)

dice_metric = DiceMetric(include_background=False, reduction='mean')