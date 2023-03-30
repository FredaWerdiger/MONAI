import sys
sys.path.append('/data/gpfs/projects/punim1086/ctp_project/MONAI/')
from monai_fns import *
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, decollate_batch, GridPatchDataset, PatchIterd
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.handlers import EarlyStopHandler
from monai.handlers.utils import from_engine
from monai.utils import first, set_determinism
# from torchmetrics import Dice
from monai.visualize import GradCAM
from sklearn.metrics import classification_report
from monai.networks.nets import UNet, AttentionUnet
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    Invertd,
    GaussianSmoothd,
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RepeatChannelD,
    Resized,
    SaveImaged,
    ScaleIntensityd,
    SqueezeDimd,
    ThresholdIntensityd,
    SplitDimd,
)

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import time
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# from numba import cuda
import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as f
from torch.optim import Adam
from models import *
from densenet import *


def define_zvalues(ct_img):
    z_min = int(ct_img.shape[2] * .25)
    z_max = int(ct_img.shape[2] * .85)

    steps = int((z_max - z_min) / 18)

    if steps == 0:
        z_min = 0
        z_max = ct_img.shape[2]
        steps = 1

    z = list(range(z_min, z_max))

    rem = int(len(z) / steps) - 18

    if rem < 0:
        add_on = [z[-1] for n in range(abs(rem))]
        z.extend(add_on)
    elif rem % 2 == 0:
        z_min = z_min + int(rem / 2 * steps) + 1
        z_max = z_max - int(rem / 2 * steps) + 1

    elif rem % 2 != 0:
        z_min = z_min + math.ceil(rem / 2)
        z_max = z_max - math.ceil(rem / 2) + 1

    z = list(range(z_min, z_max, steps))

    if len(z) == 19:
        z = z[1:]
    elif len(z) == 20:
        z = z[1:]
        z = z[:18]

    return z


def create_dwi_ctp_proba_map(dwi_ct_img,
                            gt,
                            proba_50,
                            proba_70,
                            proba_90,
                            savefile,
                            z,
                            ext='png',
                            save=False,
                            dpi=250):
    dwi_ct_img, gt, proba_50, proba_70, proba_90 = [np.rot90(im) for im in [dwi_ct_img, gt, proba_50, proba_70, proba_90]]
    dwi_ct_img, gt, proba_50, proba_70, proba_90 = [np.fliplr(im) for im in [dwi_ct_img, gt, proba_50, proba_70, proba_90]]
    proba_50_mask = proba_50 == 0
    proba_70_mask = proba_70 == 0
    proba_90_mask = proba_90 == 0
    masked_dwi = np.ma.array(dwi_ct_img, mask=~proba_50_mask)
    gt_mask = gt == 0
    masked_dwi_gt = np.ma.array(dwi_ct_img, mask=~gt_mask)
    proba_50 = np.where(proba_50 == 0, np.nan, proba_50)
    proba_70 = np.where(proba_70 == 0, np.nan, proba_70)
    proba_90 = np.where(proba_90 == 0, np.nan, proba_90)
    gt = np.where(gt == 0, np.nan, gt)


    fig, axs = plt.subplots(6, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.1, wspace=-0.3)
    axs = axs.ravel()
    for ax in axs:
        ax.axis("off")
    for i in range(6):
        print(i)

        axs[i].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        axs[i].imshow(gt[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)
        axs[i+6].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        axs[i+6].imshow(proba_50[:, :, z[i]], cmap='gnuplot',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
        axs[i+6].imshow(proba_70[:, :, z[i]], cmap='Wistia',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
        axs[i+6].imshow(proba_90[:, :, z[i]], cmap='bwr',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)

    if 12 > len(z):
        max2 = len(z)
    else:
        max2 = 12
    for i in range(6, max2):
        print(i)
        axs[i + 6].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        axs[i + 6].imshow(gt[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)
        axs[i+12].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        im = axs[i+12].imshow(proba_50[:, :, z[i]], cmap='gnuplot',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
        axs[i+12].imshow(proba_70[:, :, z[i]], cmap='Wistia',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
        axs[i+12].imshow(proba_90[:, :, z[i]], cmap='bwr',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
    if not 12 > len(z):
        if len(z) > 18:
            max3 = 18
        else:
            max3 = len(z)
        for i in range(12, max3):
            print(i)
            axs[i + 12].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                              interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
            axs[i + 12].imshow(gt[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)
            axs[i + 18].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                               interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
            im = axs[i + 18].imshow(proba_50[:, :, z[i]], cmap='gnuplot',
                                    interpolation='hanning', alpha=1, vmin=0, vmax=1)
            axs[i + 18].imshow(proba_70[:, :, z[i]], cmap='Wistia',
                               interpolation='hanning', alpha=1, vmin=0, vmax=1)
            axs[i + 18].imshow(proba_90[:, :, z[i]], cmap='bwr',
                               interpolation='hanning', alpha=1, vmin=0, vmax=1)

    if savefile:
        plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
        plt.close()


HOMEDIR = os.path.expanduser('~/')
if os.path.exists(HOMEDIR + 'mediaflux/'):
    directory = HOMEDIR + 'mediaflux/data_freda/ctp_project/CTP_DL_Data/'
    ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                            usecols=['subject', 'segmentation_type', 'dl_id'])
if os.path.exists('Z:/data_freda'):
    directory = 'Z:/data_freda/ctp_project/CTP_DL_Data/'
    ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                            usecols=['subject', 'segmentation_type', 'dl_id'])

elif os.path.exists('/data/gpfs/projects/punim1086/ctp_project'):
    directory = '/data/gpfs/projects/punim1086/ctp_project/CTP_DL_Data/'
    ctp_dl_df = pd.read_csv('/data/gpfs/projects/punim1086/study_design/study_lists/data_for_ctp_dl.csv',
                            usecols=['subject', 'segmentation_type', 'dl_id'])
elif os.path.exists('/media/mbcneuro'):
    directory = '/media/mbcneuro/CTP_DL_Data/'
    ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                            usecols=['subject', 'segmentation_type', 'dl_id'])
elif os.path.exists('/media/fwerdiger'):
    directory = '/media/fwerdiger/Storage/CTP_DL_Data/'
    ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                            usecols=['subject', 'segmentation_type', 'dl_id'])
elif os.path.exists('D:/ctp_project_data/CTP_DL_Data/'):
    directory = 'D:/ctp_project_data/CTP_DL_Data/'
    ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                            usecols=['subject', 'segmentation_type', 'dl_id'])

# train/validation/test split

data_dir = os.path.join(directory, 'DATA')
isles_dir = os.path.join(directory, 'ISLES')

image_paths = []
mask_paths = []
ncct_paths = []

for loc in [data_dir, isles_dir]:
    image_paths.extend(glob.glob(os.path.join(loc, 'images', '*')))
    mask_paths.extend(glob.glob(os.path.join(loc, 'masks', '*')))
    ncct_paths.extend(glob.glob(os.path.join(loc, 'ncct', '*')))

image_paths.sort()
mask_paths.sort()
ncct_paths.sort()

# have to retrieve slices that have lesions in them
# have to do train/test/split on the patches

def make_dict(id):
    id = [str(num).zfill(3) for num in id]
    paths1 = [file for file in image_paths
              if file.split('.nii.gz')[0].split('_')[-1] in id]
    paths2 = [file for file in ncct_paths if file.split('.nii.gz')[0].split('_')[-1] in id]
    paths3 = [file for file in mask_paths if file.split('.nii.gz')[0].split('_')[-1] in id]

    files_dict = [{"image": image_name, "ncct": ncct_name, "label": label_name} for image_name, ncct_name, label_name in
                  zip(paths1, paths2, paths3)]

    return files_dict

image_files = make_dict(ctp_dl_df.dl_id.to_list())


image_size = [128]
batch_size = 64
max_epochs = 300
val_interval = 2
out_tag = 'slice'
# feature order = ['DT', 'CBF', 'CBV', 'MTT', 'ncct']
features = ['DT', 'CBF', 'ncct']
features_transform = ['image_' + string for string in [feature for feature in features if "ncct" not in feature]]
if 'ncct' in features:
    features_transform += ['ncct']
features_string = ''
for feature in features:
    features_string += '_'
    features_string += feature

set_determinism(seed=42)

data_transforms = Compose(
    [
        LoadImaged(keys=["image", "ncct", "label"]),
        EnsureChannelFirstd(keys=["image", "ncct", "label"]),
        # Resized(keys=["image", "ncct", "label"],
        #         mode=['trilinear', 'trilinear', "nearest"],
        #         align_corners=[True, True, None],
        #         spatial_size=image_size * 3),
        SplitDimd(keys="image", dim=0, keepdim=True,
                  output_postfixes=['DT', 'CBF', 'CBV', 'MTT']),
        # SaveImaged(keys="ncct",
        #            output_dir=transform_dir,
        #            meta_keys="ncct_meta_dict",
        #            output_postfix="transform",
        #            resample=False,
        #            separate_folder=False),
        ConcatItemsd(keys=features_transform, name="image", dim=0),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=1.0),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
        EnsureTyped(keys=["image", "label"]),
    ]
)

volume_ds = CacheDataset(
    data=image_files,
    transform=data_transforms,

)

patch_transform = Compose(
    [
        SqueezeDimd(keys=["image", "label"], dim=-1),  # squeeze the last dim
        Resized(keys=["label", "label"], spatial_size=image_size*2,
                mode=['trilinear', "nearest"],
                align_corners=[True, None],
                ),
        # to use crop/pad instead of reszie:
        # ResizeWithPadOrCropd(keys=["img", "seg"], spatial_size=[48, 48], mode="replicate"),
    ]
)

patch_func = PatchIterd(
    keys=["image", "label"],
    patch_size=(None, None, 1),  # dynamic first two dimensions
    start_pos=(0, 0, 0)
)

patch_ds = GridPatchDataset(
    data=volume_ds,
    patch_iter=patch_func,
    transform=patch_transform,
    with_coordinates=False)

data_loader = DataLoader(
    patch_ds,
    batch_size=1,
    pin_memory=torch.cuda.is_available(),
)

lesion_slices = []
ids = []
id = 0 # build ID
sizes = []
for i, data_item in enumerate(list(data_loader)):
    label = data_item["label"]
    lesion_size = np.count_nonzero(label.numpy())
    x, y = data_item['label_meta_dict']['pixdim'][0][1:3].numpy()
    voxel_size = (x * y) /100
    lesion_size = lesion_size * voxel_size
    if lesion_size >0:
        lesion_slices.append(data_item)
        ids.append(id)
        id += 1
        sizes.append(lesion_size)

sizes_bin = (np.asarray(sizes) < 0.05) * 1

df = pd.DataFrame(columns=['ids', 'sizes'])
df['ids'] = ids
df['sizes'] = sizes_bin
num_train = int(np.ceil(0.6 * len(lesion_slices)))
num_validation = int(np.ceil(0.2 * len(lesion_slices)))
num_test = len(lesion_slices) - (num_train + num_validation)

random_state = 42

train_id, test_id = train_test_split(ids,
                                     train_size=num_train,
                                     test_size=num_test+num_validation,
                                     random_state=random_state,
                                     shuffle=True,
                                     stratify=sizes_bin)

# get labels list that correspond with test id
test_df = df[df.apply(lambda x: x['ids'] in test_id, axis=1)]
test_labels = test_df.sizes.to_list()
test_id = test_df.ids.to_list()

validation_id, test_id = train_test_split(test_id,
                                          train_size=num_validation,
                                          test_size=num_test,
                                          random_state=random_state,
                                          shuffle=True,
                                          stratify=test_labels)


train_files = []
test_files = []
validation_files = []

for i, im in enumerate(lesion_slices):
    if i in train_id:
        train_files.append(im)
    elif i in validation_id:
        validation_files.append(im)
    elif i in test_id:
        test_files.append(im)

train_ds = CacheDataset(data=train_files, transform=None)
validation_ds = CacheDataset(data=validation_files, transform=None)
test_ds = CacheDataset(data=test_files, transform=None)


train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True
)


val_loader = DataLoader(
    validation_ds,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True
)


test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True
)

ch_in = train_ds[0]["image"].shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channels = (16, 32, 64)

model = AttentionUnet(
    spatial_dims=2,
    in_channels=ch_in,
    out_channels=2,
    channels=(16, 32, 64),
    strides=(2, 2),
).to(device)

loss_function = DiceCELoss(smooth_dr=1e-5,
                           smooth_nr=0,
                           to_onehot_y=True,
                           softmax=True,
                           include_background=False,
                           squared_pred=True,
                           lambda_dice=1,
                           lambda_ce=1)

optimizer = Adam(model.parameters(),
                 1e-4,
                 weight_decay=1e-5)

dice_metric = DiceMetric(include_background=False, reduction='mean')

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
epoch_loss_values = []
dice_metric_values = []
dice_metric_values_train = []
best_metric = -1
best_metric_epoch = -1

post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
start = time.time()
model_path = 'best_metric_' + model._get_name() + '_' + str(max_epochs) + '_' + features_string +'.pth'


for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    epoch_loss = 0
    step = 0
    model.train()
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
        epoch_loss += loss.item()
        optimizer.step()
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        print("Evaluating...")
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = model(val_inputs)

                # compute metric for current iteration
                # dice_metric_torch_macro(val_outputs, val_labels.long())
                # now to for the MONAI dice metric
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(val_outputs, val_labels)

            mean_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            dice_metric_values.append(mean_dice)

            if mean_dice > best_metric:
                best_metric = mean_dice
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    directory, 'out_' + out_tag, model_path))
                print("saved new best metric model")

            print(
                f"current epoch: {epoch + 1} current mean dice: {mean_dice:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
    del loss, outputs
end = time.time()
time_taken = end - start
print(f"Time taken: {round(time_taken, 0)} seconds")
time_taken_hours = time_taken / 3600
time_taken_mins = np.ceil((time_taken / 3600 - int(time_taken / 3600)) * 60)
time_taken_hours = int(time_taken_hours)

model_name = model._get_name()
loss_name = loss_function._get_name()

with open(
        directory + 'out_' + out_tag + '/model_info_' + str(
            max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_' + features_string + '.txt',
        'w') as myfile:
    myfile.write(f'Train dataset size: {len(train_files)}\n')
    myfile.write(f'Validation dataset size: {len(validation_files)}\n')
    myfile.write(f'Test dataset size: {len(test_files)}\n')
    myfile.write(f'Intended number of features: {len(features)}\n')
    myfile.write(f'Actual number of features: {ch_in}\n')
    myfile.write('Features: ')
    myfile.write(features_string)
    myfile.write('\n')
    myfile.write(f'Model: {model_name}\n')
    myfile.write(f'Loss function: {loss_name}\n')
    myfile.write(f'Number of epochs: {max_epochs}\n')
    myfile.write(f'Batch size: {batch_size}\n')
    myfile.write(f'Image size: {image_size}\n')
    myfile.write(f'channels: {channels}\n')
    myfile.write(f'Validation interval: {val_interval}\n')
    myfile.write(f"Best metric: {best_metric:.4f}\n")
    myfile.write(f"Best metric epoch: {best_metric_epoch}\n")
    myfile.write(f"Time taken: {time_taken_hours} hours, {time_taken_mins} mins\n")

    # plot things
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Average Loss per Epoch")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Mean Dice (Accuracy)")
x = [val_interval * (i + 1) for i in range(len(dice_metric_values))]
y = dice_metric_values
plt.xlabel("epoch")
plt.plot(x, y, 'b', label="Dice on validation data")
y = dice_metric_values_train
plt.plot(x, y, 'k', label="Dice on training data")
plt.legend(loc="center right")
plt.savefig(os.path.join(directory + 'out_' + out_tag,
                         'loss_plot_' + str(max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_'
                         + features_string + '.png'),
            bbox_inches='tight', dpi=300, format='png')
plt.close()

