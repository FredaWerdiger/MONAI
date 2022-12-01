# this code is to get a probability map out instead of a binary mask
# currently in piece
from monai_fns import *
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
import glob
from collections import OrderedDict
from monai.data import Dataset, DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet, UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    Resized,
    SaveImaged,

)
import torch
import torch.nn.functional as f

def main(root_dir, ctp_df, model_path, out_tag, ddp=False):

    device = 'cuda'
    # test on external data
    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            Resized(keys="image",
                    mode='trilinear',
                    align_corners=True,
                    spatial_size=(128, 128, 128)),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    test_files = BuildDataset(root_dir, 'test').no_seg_dict
    test_ds = Dataset(
        data=test_files, transform=test_transforms)

    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

    post_transforms = Compose([
        EnsureTyped(keys=["pred"]),
        EnsureChannelFirstd(keys="label"),
        Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        )
    ])

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs = test_data["image"].to(device)
            roi_size = (64, 64, 64)
            sw_batch_size = 2
            test_data["pred"] = sliding_window_inference(
                test_inputs, roi_size, sw_batch_size, model)
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
            test_output, test_image = from_engine(["pred", "image"])(test_data)
            prob = f.softmax(test_output, dim=0)

            original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])
            volx, voly, volz = original_image[1]['pixdim'][1:4] # meta data
            pixel_vol = volx * voly * volz