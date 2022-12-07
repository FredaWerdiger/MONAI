import matplotlib.pyplot as plt
import torch

import monai_fns
from monai_fns import *
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.visualize import GradCAM, blend_images
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    Rotate90d,
    ScaleIntensityd,
    EnsureChannelFirstd,
    EnsureTyped,
    Flipd,
    NormalizeIntensityd,
    Resized
    )

model =  UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=2,
        channels=(16, 32, 64),
        strides=(2, 2),
        num_res_units=2,
        norm=Norm.BATCH
).cuda()

log = model.load_state_dict(ddp_state_dict('D:/ctp_project_data/CTP_DL_Data/out_unet_ddp_no_patch/best_metric_model600.pth'))

cam = GradCAM(nn_module=model, target_layers='model.2.0.adn.A')

trans = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys="image"),
            Resized(keys="image",
                    mode='trilinear',
                    align_corners=True,
                    spatial_size=(32, 32, 32)),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )

test_files = BuildDataset('D:/ctp_project_data/CTP_DL_Data', 'test').no_seg_dict
for data in trans(test_files[:1]):
    cam_result = cam(x=data['image'].unsqueeze(0).cuda())

img = data['image'].cpu()
heatmap = cam_result.squeeze(0).cpu()

blend_img = blend_images(image=img, label=heatmap, alpha=0.5, cmap="hsv", rescale_arrays=False)
blend_img = blend_img.moveaxis(0,-1)

fig, axes = plt.subplots(1, 3, figsize=(10, 10), facecolor='white')

im_0 = axes[0].imshow(img.squeeze(), cmap='gray')
axes[0].axis('off')
fig.colorbar(im_0, ax=axes[0], shrink=0.25)

im_1 = axes[1].imshow(heatmap.squeeze(), cmap='jet')
axes[1].axis('off')
fig.colorbar(im_1, ax=axes[1], shrink=0.25)

im_2 = axes[2].imshow(blend_img)
axes[2].axis('off')
fig.colorbar(im_2, ax=axes[2], shrink=0.25)

plt.savefig('out.png')