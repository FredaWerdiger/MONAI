import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import tempfile
import glob
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.layers import Norm
from monai.transforms import (
    AsDiscreted,
    Compose,
    Invertd,
    LoadImage,
    LoadImaged,
    CropForegroundd,
    NormalizeIntensityd,
    ScaleIntensityRangePercentilesd,
    EnsureChannelFirstd,
    EnsureTyped,
    Resized,
    SaveImaged
)
import pandas as pd
import torch
import os

def make_dict(root, string):
    images = sorted(
        glob.glob(os.path.join(root, string, 'images', '*.nii.gz'))
    )
    # labels = sorted(
    #     glob.glob(os.path.join(root, string, 'masks', '*.nii.gz'))
    # )
    return [
        {"image": image_name}
        for image_name in images
    ]


if os.path.exists('/media/'):
    directory = '/media/mbcneuro/HDD1/DWI_Training_Data_reshuffle/'
    ctp_df = pd.read_csv(
        '/home/mbcneuro/PycharmProjects/study_design/study_lists/dwi_inspire_dl.csv',
        index_col='dl_id'
    )

elif os.path.exists('D:'):
    directory = 'D:/ctp_project_data/DWI_Training_Data_reshuffle/'

root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

test_files = make_dict(root_dir, "no_seg")
# replace with path
model_file = "D:/ctp_project_data/DWI_Training_Data_INSP/out_scale_1_99/best_metric_model100.pth"
out_dir = root_dir + "no_seg/images"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# test on external data
test_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        CropForegroundd(keys="image", source_key="image"),
        Resized(keys="image",
                mode='trilinear',
                align_corners=True,
                spatial_size=(128, 128, 128)),
        ScaleIntensityRangePercentilesd(keys="image",
                                        lower=1,
                                        upper=99,
                                        b_min=0.0,
                                        b_max=10.0,
                                        channel_wise=True,
                                        clip=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys="image"),
        SaveImaged(keys="image",
                   output_dir=out_dir,
                   output_postfix="transform",
                   resample=False,
                   separate_folder=False)
    ]
)


test_ds = Dataset(
    data=test_files, transform=test_transforms)

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
    AsDiscreted(keys="pred",
                argmax=True),
    SaveImaged(keys="pred",
               meta_keys="pred_meta_dict",
               output_dir=out_dir,
               output_postfix="seg", resample=False, separate_folder=False),
])

device = torch.device("cuda:0")
model = UNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=2,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

loader = LoadImage()

model.load_state_dict(torch.load(model_file))
model.eval()


with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        test_inputs = test_data["image"].to(device)
        roi_size = (64, 64, 64)
        sw_batch_size = 4
        test_data["pred"] = sliding_window_inference(
            test_inputs, roi_size, sw_batch_size, model)

        test_data = [post_transforms(i) for i in decollate_batch(test_data)]




