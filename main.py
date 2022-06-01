# following tutorial from BRATs segmentation
# two classes insead of 4 classes
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import math
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
from sklearn.metrics import f1_score
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
CropForegroundd,
RandAffined,
ScaleIntensityRanged,
MapTransform,
NormalizeIntensityd,
ScaleIntensityRangePercentilesd,
Orientationd,
RandFlipd,
AddChanneld,
RandAdjustContrastd,
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

import pandas as pd
import torch
from jinja2 import Environment, FileSystemLoader
import os
import sys
from termcolor import colored

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

#free_gpu_cache()

print_config()


# for creating images
def define_dvalues(dwi_img):
    steps = int(dwi_img.shape[2]/18)
    rem = int(dwi_img.shape[2]/steps)-18

    if rem % 2 == 0:
        d_min = 0 + int(rem/2*steps) + 1
        d_max = dwi_img.shape[2] - int(rem/2*steps) + 1

    elif rem % 2 != 0:
        d_min = 0 + math.ceil(rem*steps/2)
        d_max = dwi_img.shape[2] - math.ceil(rem/2*steps) + 1

    d = range(d_min, d_max, steps)

    if len(d) == 19:

        d = d[1:]
    return d


def define_dvalues_big(dwi_img):
    dwi_img_small = dwi_img[10:120]
    steps = int(dwi_img_small.shape[0]/18)
    rem = int(dwi_img_small.shape[0]/steps)-18

    if rem % 2 == 0:
        d_min = 0 + int(rem/2*steps) + 1
        d_max = dwi_img_small.shape[0] - int(rem/2*steps)

    elif rem % 2 != 0:
        d_min = 0 + math.ceil(rem*steps/2)
        d_max = dwi_img_small.shape[0] - math.ceil(rem/2*steps)

    d = range(d_min + 10, d_max + 10, steps)

    if len(d) == 19:
        d = range(d_min + steps + 10, d_max + 10, steps)
    return d


def create_mrlesion_img(dwi_img, dwi_lesion_img, savefile, d, ext='png', dpi=250):
    dwi_lesion_img = np.rot90(dwi_lesion_img)
    dwi_img = np.rot90(dwi_img)
    mask = dwi_lesion_img < 1
    masked_im = np.ma.array(dwi_img, mask=~mask)

    fig, axs = plt.subplots(3, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.6, wspace=-0.1)

    axs = axs.ravel()

    for i in range(len(d)):
        axs[i].imshow(dwi_lesion_img[:, :,d[i]], cmap='Wistia', vmin=0.5, vmax=1)
        axs[i].imshow(masked_im[:, :, d[i]], cmap='gray', interpolation='hanning', vmin=0, vmax=300)
        axs[i].axis('off')
    # plt.show()
    plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
    plt.close()


def create_mr_img(dwi_img, savefile, d, ext='png', dpi=250):
    dwi_img = np.rot90(dwi_img)
    fig, axs = plt.subplots(3, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.6, wspace=-0.1)
    axs = axs.ravel()
    for i in range(len(d)):
        axs[i].imshow(dwi_img[:, :, d[i]], cmap='gray', interpolation='hanning', vmin=0, vmax=300)
        axs[i].axis('off')
    # plt.show()
    plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
    plt.close()


def create_mr_big_img(dwi_img, savefile, d, ext='png', dpi=250):
    dwi_img = np.rot90(dwi_img)
    fig, axs = plt.subplots(3, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.6, wspace=-0.1)
    axs = axs.ravel()
    for i in range(len(d)):
        axs[i].imshow(dwi_img[:, :, d[i]], cmap='gray')
        axs[i].axis('off')
    # plt.show()
    plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
    plt.close()


def create_adc_img(dwi_img, savefile, d, ext='png', dpi=250):
    dwi_img = np.rot90(dwi_img)
    fig, axs = plt.subplots(3, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.6, wspace=-0.1)
    axs = axs.ravel()
    for i in range(len(d)):
        axs[i].imshow(dwi_img[:, :, d[i]], cmap='gray', interpolation='hanning', vmin=0, vmax=1500)
        axs[i].axis('off')
    #plt.show()
    plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
    plt.close()


def create_paper_img(dwi_img, gt, pred, savefile, d, ext='png', dpi=250):
    dwi_img, gt, pred = [np.rot90(im) for im in [dwi_img, gt, pred]]
    lesions = gt + pred
    mask = lesions == 0
    masked_img = np.ma.array(dwi_img, mask=~mask)

    false_neg = np.ma.masked_where(
        np.logical_and(pred == 0, gt == 1), gt) * gt
    true_pos = np.ma.masked_where(np.logical_and(pred == 1, gt == 1),
                                  gt) * gt
    false_pos = np.ma.masked_where(
        np.logical_and(pred == 1, gt == 0), pred) * pred

    fig, axs = plt.subplots(3, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.6, wspace=-0.1)
    axs = axs.ravel()
    for i in range(len(d)):
        axs[i].imshow(false_neg[:,:,d[i]], cmap='gist_rainbow', vmin=0, vmax=1)
        axs[i].imshow(false_pos[:, :, d[i]], cmap='brg', vmin=0, vmax=1)
        # axs[i].imshow(true_pos[:,:,d[i]], cmap='tab10', vmin=0, vmax=1)
        axs[i].imshow(masked_img[:,:,d[i]], cmap='gray', interpolation='hanning', vmin=0, vmax=300)
        axs[i].axis('off')
    # plt.show()
    plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
    plt.close()

def create_overviewhtml(subject_id, df, outdir):
    '''
    Function that creates a HTML file with clinical information and imaging.
    '''
    templateLoader = FileSystemLoader(
        searchpath='./')
    templateEnv = Environment(loader=templateLoader)
    TEMPLATE_FILE = "template.html"
    template = templateEnv.get_template(TEMPLATE_FILE)

    savefolder = outdir + 'htmls'
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    ptData = df.set_index('id').loc[subject_id, :]
    treat = ptData['Treatment type(0=no treatment,1=iv only, 2=IA only, 3= Both ia +iv, 4=iv only IA planned but not delivery,5=no information)']
    if treat == 0:
        treat = "No treatment"
    elif treat == 1:
        treat = "IV only"
    elif treat == 2:
        treat = "IA only"
    elif treat == 3:
        treat = "IV and IA"

    small = ptData['small lesion']
    small = 'yes' if small == 1 else 'no'
    thickness = ptData['dwi_slice_thickness']
    make = str(ptData['dwi_Manufacturer']) + ' ' + str(ptData['dwi_Model'])
    day = ptData['day']
    cause = ptData['Stroke Mechanism']

    output = template.render(subject_id=subject_id,
                             inspire_id=ptData['subject'],
                             dice=round(ptData['dice'], 4),
                             folder=outdir + 'images',
                             treatment=treat,
                             make=make,
                             thickness=thickness,
                             small=small,
                             day=day,
                             cause=cause
                             )
    # populate template

    # save html file

    with open(os.path.join(savefolder, subject_id + '.html'), 'w') as f:
        f.write(output)


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


if os.path.exists('/media/'):
    directory = '/media/mbcneuro/HDD1/DWI_Training_Data_INSP/'
    ctp_df = pd.read_csv(
        '/home/mbcneuro/PycharmProjects/study_design/study_lists/dwi_inspire_dl.csv',
        index_col='dl_id'
    )


elif os.path.exists('D:'):
    directory = 'D:/ctp_project_data/DWI_Training_Data/'
    ctp_df = pd.read_csv(
        'C:/Users/fwerdiger/PycharmProjects/study_design/study_lists/dwi_inspire_dl.csv',
        index_col='dl_id')


root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


train_files, val_files, test_files = [
    make_dict(root_dir, string) for string in ['train', 'validation', 'test']]

set_determinism(seed=42)

# test different transforms


out_tag = "final"
max_epochs = 600
# create outdir
if not os.path.exists(root_dir + 'out_' + out_tag):
    os.makedirs(root_dir + 'out_' + out_tag)

train_transforms = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"],
                mode=['trilinear', "nearest"],
                align_corners=[True, None],
                spatial_size=(128, 128, 128)),
        ScaleIntensityRangePercentilesd(keys="image",
                                        lower=1,
                                        upper=99,
                                        b_min=0.0,
                                        b_max=10.0,
                                        channel_wise=True,
                                        clip=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        RandAdjustContrastd(keys="image", prob=1, gamma=(0.5, 1)),
        EnsureTyped(keys=["image", "label"]),
    ]
)


val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"],
                mode=['trilinear', "nearest"],
                align_corners=[True, None],
                spatial_size=(128, 128, 128)),
        ScaleIntensityRangePercentilesd(keys="image",
                                        lower=1,
                                        upper=99,
                                        b_min=0.0,
                                        b_max=10.0,
                                        channel_wise=True,
                                        clip=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
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

val_loader = DataLoader(
    val_ds,
    batch_size=2,
    shuffle=False,
    num_workers=4)

# Uncomment to display data

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
model = UNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=2,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
# model = SegResNet(
#     blocks_down=[1, 2, 2, 4],
#     blocks_up=[1, 1, 1],
#     init_filters=8,
#     in_channels=2,
#     out_channels=2,
#     dropout_prob=0.2,
# ).to(device)
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=False, reduction="mean")

val_interval = 1
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
        # commenting out print function
        # print(
        #     f"{step}/{len(train_ds) // train_loader.batch_size}, "
        #     f"train_loss: {loss.item():.4f}")
    #lr_scheduler.step()
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
                # unsure how to optimize this
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
# generate loss plot
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
plt.savefig(os.path.join(root_dir + 'out_' + out_tag, model_name.split('.')[0] + 'plot_loss.png'),
            bbox_inches='tight', dpi=300, format='png')
plt.show()

# evaluate during training process
# model.load_state_dict(torch.load(
#     os.path.join(root_dir, model_name)))
# model.eval()
# with torch.no_grad():
#     for i, val_data in enumerate(val_loader):
#         roi_size = (128, 128, 128)
#         sw_batch_size = 1
#         val_outputs = sliding_window_inference(
#             val_data["image"].to(device), roi_size, sw_batch_size, model
#         )
#         # plot the slice [:, :, 80]
#         plt.figure("check", (18, 6))
#         plt.subplot(1, 3, 1)
#         plt.title(f"image {i}")
#         plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
#         plt.subplot(1, 3, 2)
#         plt.title(f"label {i}")
#         plt.imshow(val_data["label"][0, 0, :, :, 80])
#         plt.subplot(1, 3, 3)
#         plt.title(f"output {i}")
#         plt.imshow(torch.argmax(
#             val_outputs, dim=1).detach().cpu()[0, :, :, 80])
#         plt.show()
#         if i == 2:
#             break

# test on external data
test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
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
        EnsureTyped(keys=["image", "label"]),
        SaveImaged(keys="image", output_dir=root_dir + "out", output_postfix="transform", resample=False)
    ]
)


test_ds = Dataset(
    data=test_files, transform=test_transforms)

test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

post_transforms = Compose([
    EnsureTyped(keys=["pred", "label"]),
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
    ),
    AsDiscreted(keys="pred", argmax=True, to_onehot=2),
    AsDiscreted(keys="label", to_onehot=2),
    SaveImaged(keys="pred",
               meta_keys="pred_meta_dict",
               output_dir=root_dir + "out_" + out_tag,
               output_postfix="seg", resample=False),
])

from monai.transforms import LoadImage, ScaleIntensityRangePercentiles, EnsureType
loader = LoadImage()
model.load_state_dict(torch.load(
    os.path.join(root_dir, 'out_' + out_tag, model_name)))

model.eval()

results = pd.DataFrame(columns=['id', 'dice', 'size'])
results['id'] = ['test_' + str(item).zfill(3) for item in range(1, len(test_loader) + 1)]

with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        test_inputs = test_data["image"].to(device)
        roi_size = (64, 64, 64)
        sw_batch_size = 4
        test_data["pred"] = sliding_window_inference(
            test_inputs, roi_size, sw_batch_size, model)

        test_data = [post_transforms(i) for i in decollate_batch(test_data)]

        # uncomment the following lines to visualize the predicted results
        test_output, test_label, test_image = from_engine(["pred", "label", "image"])(test_data)

        # get original image, and normalize it so we can see the normalized image
        # this is both channels
        original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])[0]
        original_adc = original_image[:, :, :, 1]
        original_image = original_image[:, :, :, 0]
        ground_truth = test_label[0][1].detach().numpy()
        prediction = test_output[0][1].detach().numpy()
        transformed_image = test_inputs[0][0].detach().cpu().numpy()
        size = prediction.sum()
        name = "test_" + os.path.basename(
            test_data[0]["image_meta_dict"]["filename_or_obj"]).split('.nii.gz')[0].split('_')[1]
        save_loc = root_dir + "out_" + out_tag + "/images/" + name + "_"

        if not os.path.exists(root_dir + "out_" + out_tag + "/images/"):
            os.makedirs(root_dir + "out_" + out_tag + "/images/")

        create_paper_img(
            original_image,
            ground_truth,
            prediction,
            save_loc + "paper.png",
            define_dvalues(original_image),
            'png',
            dpi=300
        )
        create_mr_img(
            original_image,
            save_loc + "dwi.png",
            define_dvalues(original_image),
            'png',
            dpi=300)
        create_adc_img(
            original_adc,
            save_loc + "adc.png",
            define_dvalues(original_image),
            'png',
            dpi=300)

        [create_mrlesion_img(
            original_image,
            im,
            save_loc + name + '.png',
            define_dvalues(original_image),
            'png',
            dpi=300) for im, name in zip([prediction, ground_truth], ["pred", "truth"])]

        create_mr_big_img(transformed_image,
                          save_loc + "dwi_tran.png",
                          define_dvalues_big(transformed_image),
                          'png',
                          dpi=300)

        # uncomment below to visualise results.
        # plt.figure("check", (24, 6))
        # plt.subplot(1, 4, 1)
        # plt.imshow(original_image[:, :, 12], cmap="gray")
        # plt.title(f"image {name}")
        # plt.subplot(1, 4, 2)
        # plt.imshow(test_image[0].detach().cpu()[0, :, :, 12], cmap="gray")
        # plt.title(f"transformed image {name}")
        # plt.subplot(1, 4, 3)
        # plt.imshow(test_label[0].detach().cpu()[:, :, 12])
        # plt.title(f"label {name}")
        # plt.subplot(1, 4, 4)
        # plt.imshow(test_output[0].detach().cpu()[1, :, :, 12])
        # plt.title(f"Dice score {dice_score}")
        # plt.show()

        a = dice_metric(y_pred=test_output, y=test_label)
        dice_score = round(a.item(), 4)
        results.loc[results.id == name, 'size'] = size
        results.loc[results.id == name, 'dice'] = dice_score

    # aggregate the final mean dice result
    metric = dice_metric.aggregate().item()
    # reset the status for next validation round
    dice_metric.reset()

print(f"Mean dice on test set: {metric}")

results['mean_dice'] = metric
results_join = results.join(
    ctp_df[~ctp_df.index.duplicated(keep='first')],
    on='id',
    how='left')
results_join.to_csv(root_dir + 'out_' + out_tag + '/results.csv', index=False)

for sub in results_join['id']:
    create_overviewhtml(sub, results_join, root_dir + 'out_' + out_tag + '/')
