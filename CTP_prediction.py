import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.utils import first, set_determinism
from torchmetrics import Dice
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    EnsureChannelFirst,
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

print_config()


#TODO: make a class for these DL related functions


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


def prepare(dataset,
            rank,
            world_size,
            batch_size,
            pin_memory=False,
            num_workers=0):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False)
    dataloader=DataLoader(dataset,
                          batch_size=batch_size,
                          pin_memory=pin_memory,
                          num_workers=num_workers,
                          drop_last=False,
                          shuffle=False,
                          sampler=sampler
                          )
    return dataloader

