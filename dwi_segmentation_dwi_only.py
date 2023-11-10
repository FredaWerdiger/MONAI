# following tutorial from BRATs segmentation
# two classes insead of 4 classes
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import math
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from sklearn.metrics import f1_score
from monai.config import print_config
from monai.data import Dataset, CacheDataset, DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet, UNet, AttentionUnet, DenseNet
from monai.networks.layers import Norm
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImage,
    LoadImaged,
    CropForegroundd,
    NormalizeIntensityd,
    ScaleIntensityRangePercentilesd,
    RandAffined,
    RandFlipd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    Resized,
    SaveImaged,
    SplitDimd
)

from monai.utils import first, set_determinism
from monai_fns import *
from densenet import *
import torch
import os

from recursive_data import *

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


def main():
    directory = '/data/gpfs/projects/punim1086/ctp_project/DWI_Training_Data/'
    # existing_model = directory + 'out_densenetFCN_batch1/learning_rate_1e4/best_metric_model600.pth'

    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)

    # create outdir
    out_tag = "densenetFCN_batch1/learning_rate_1e4/with_isles"
    if not os.path.exists(root_dir + 'out_' + out_tag):
        os.makedirs(root_dir + 'out_' + out_tag)

    train_files, val_files, test_files = [
        make_dict(root_dir, string) for string in ['train', 'validation', 'test']]
    semi_files = get_semi_dataset()
    train_files = semi_files + train_files

    corrections = get_corrections()
    isles = BuildDataset(root_dir, 'ISLES22').images_dict[:80]
    print(f"Number of corrections added: {len(corrections)}")
    train_files = train_files + semi_files + corrections + isles

    set_determinism(seed=42)

    max_epochs = 600
    batch_size = 1
    image_size = (128, 128, 128)
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            SplitDimd(keys="image", dim=0, keepdim=True,
                      output_postfixes=['b1000', 'adc']),
            Resized(keys=["image", "image_b1000", "image_adc", "label"],
                    mode=['trilinear', 'trilinear', 'trilinear', "nearest"],
                    align_corners=[True, True, True, None],
                    spatial_size=image_size),
            NormalizeIntensityd(keys="image_b1000", nonzero=True, channel_wise=True),
            RandAffined(keys=['image_b1000', 'label'], prob=0.5, translate_range=10),
            RandFlipd(keys=["image_b1000", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image_b1000", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image_b1000", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=["image_b1000"], factors=0.1, prob=1.0),
            RandShiftIntensityd(keys=["image_b1000"], offsets=0.1, prob=1.0),
            # RandAdjustContrastd(keys="image", prob=1, gamma=(0.5, 1)),
            EnsureTyped(keys=["image_b1000", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            SplitDimd(keys="image", dim=0, keepdim=True,
                      output_postfixes=['b1000', 'adc']),
            Resized(keys=["image", "image_b1000", "image_adc", "label"],
                    mode=['trilinear', 'trilinear', 'trilinear', "nearest"],
                    align_corners=[True, True, True, None],
                    spatial_size=image_size),
            NormalizeIntensityd(keys="image_b1000", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image_b1000", "label"]),
        ]
    )

    # here we don't cache any data in case out of memory issue
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=4
    )


    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=4)

    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            pin_memory=True)

    rank = 'cuda'

    model = DenseNetFCN(
        ch_in=2,
        ch_out_init=48,
        num_classes=2,
        growth_rate=16,
        layers=(4, 5, 7, 10, 12),
        bottleneck=True,
        bottleneck_layer=15
    ).to(rank)

    loss_function = DiceLoss(
        smooth_nr=0,
        smooth_dr=1e-5,
        to_onehot_y=True,
        softmax=True,
        include_background=False)
    learning_rate = 1e-4
    weight_decay = 1e-4
    optimizer = torch.optim.Adam(
        model.parameters(),
        learning_rate,
        weight_decay=weight_decay)

    #
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    val_interval = 2
    # only doing these for master node
    epoch_loss_values = []
    metric_values = []
    best_metric = -1
    best_metric_epoch = -1
    f1_mean_values = []
    # Below not needed for torchmetrics metric
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    start = time.time()
    model_name = 'best_metric_model' + str(max_epochs) + '.pth'

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        epoch_loss = 0
        step = 0
        model.train()
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image_b1000"].to(rank),
                batch_data["label"].to(rank),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            # commenting out print function
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}")
        # lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            print("Evaluating...")
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image_b1000"].to(rank),
                        val_data["label"].to(rank),
                    )
                    # unsure how to optimize this
                    val_outputs = model(val_inputs)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(val_outputs, val_labels)
                    # validate with f1 score

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
    time_taken_hours = time_taken / 3600
    time_taken_mins = np.ceil((time_taken / 3600 - int(time_taken / 3600)) * 60)
    time_taken_hours = int(time_taken_hours)

    model_name = model._get_name()
    loss_name = loss_function._get_name()
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
    plt.savefig(os.path.join(root_dir + 'out_' + out_tag,
                             str(max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_plot_loss.png'),
                bbox_inches='tight', dpi=300, format='png')
    plt.close()

    # save model results in a separate file
    with open(root_dir + 'out_' + out_tag + '/model_info_' + str(max_epochs) + '_epoch_' + model_name + '_' + loss_name + '.txt', 'w') as myfile:
        myfile.write(f'Train dataset size: {len(train_files)}\n')
        myfile.write(f'Semi-automated segmentations: {len(semi_files)}\n')
        myfile.write(f'corrected segmentations: {len(corrections)}\n')
        myfile.write(f'isles datya: {len(isles)}\n')
        myfile.write(f'Validation dataset size: {len(val_files)}\n')
        myfile.write(f'Model: {model_name}\n')
        myfile.write(f'Loss function: {loss_name}\n')
        myfile.write(f'Number of epochs: {max_epochs}\n')
        myfile.write(f'Initial learning rate: {learning_rate}\n')
        myfile.write(f'Weight decay: {weight_decay}\n')
        myfile.write(f'Batch size: {batch_size}\n')
        myfile.write(f'Image size: {image_size}\n')
        myfile.write(f'Validation interval: {val_interval}\n')
        myfile.write(f"Best metric: {best_metric:.4f}\n")
        myfile.write(f"Best metric epoch: {best_metric_epoch}\n")
        myfile.write(f"Time taken: {time_taken_hours} hours, {time_taken_mins} mins\n")

   # test on external data
    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            SplitDimd(keys="image", dim=0, keepdim=True,
                      output_postfixes=['b1000', 'adc']),
            Resized(keys=["image", "image_b1000", "image_adc"],
                    mode='trilinear',
                    align_corners=True,
                    spatial_size=(128, 128, 128)),
            NormalizeIntensityd(keys="image_b1000", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image_b1000", "label"]),
            # SaveImaged(keys="image", output_dir=root_dir + "out", output_postfix="transform", resample=False)
        ]
    )

    test_files = make_dict(root_dir, 'test')
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
                   output_dir=root_dir + "out_" + out_tag + '/pred',
                   output_postfix="pred", resample=False,
                   separate_folder=False),
    ])

    if not os.path.exists(root_dir + "out_" + out_tag + '/pred'):
        os.makedirs(root_dir + "out_" + out_tag + '/pred')

    # removing sync on step as we are running on master node
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    loader = LoadImage(image_only=False)

    model.eval()

    results = pd.DataFrame(columns=['id', 'dice', 'size', 'px_x', 'px_y', 'px_z', 'size_ml'])
    results['id'] = ['test_' + str(item).zfill(3) for item in range(1, len(test_loader) + 1)]

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs = test_data["image_b1000"].to(rank)

            test_data["pred"] = model(test_inputs)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

            test_output, test_label, test_image = from_engine(["pred", "label", "image_b1000"])(test_data)

            a = dice_metric(y_pred=test_output, y=test_label)

            dice_score = round(a.item(), 4)
            print(f"Dice score for image: {dice_score:.4f}")

            # get original image, and normalize it so we can see the normalized image
            # this is both channels
            original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])
            volx, voly, volz = original_image[1]['pixdim'][1:4] # meta data
            pixel_vol = volx * voly * volz

            original_image = original_image[0] # image data
            original_adc = original_image[:, :, :, 1]
            original_image = original_image[:, :, :, 0]
            ground_truth = test_label[0][1].detach().numpy()
            prediction = test_output[0][1].detach().numpy()
            transformed_image = test_inputs[0][0].detach().cpu().numpy()
            size = ground_truth.sum()
            size_ml = size * pixel_vol / 1000
            size_pred = prediction.sum()
            size_pred_ml = size_pred * pixel_vol / 1000
            name = "test_" + os.path.basename(
                test_data[0]["image_meta_dict"]["filename_or_obj"]).split('.nii.gz')[0].split('_')[1]
            save_loc = root_dir + "out_" + out_tag + "/images/" + name + "_"

            if not os.path.exists(root_dir + "out_" + out_tag + "/images/"):
                os.makedirs(root_dir + "out_" + out_tag + "/images/")

            # create_paper_img(
            #     original_image,
            #     ground_truth,
            #     prediction,
            #     save_loc + "paper.png",
            #     define_dvalues(original_image),
            #     'png',
            #     dpi=300
            # )
            # create_mr_img(
            #     original_image,
            #     save_loc + "dwi.png",
            #     define_dvalues(original_image),
            #     'png',
            #     dpi=300)
            # create_adc_img(
            #     original_adc,
            #     save_loc + "adc.png",
            #     define_dvalues(original_image),
            #     'png',
            #     dpi=300)
            #
            # [create_mrlesion_img(
            #     original_image,
            #     im,
            #     save_loc + name + '.png',
            #     define_dvalues(original_image),
            #     'png',
            #     dpi=300) for im, name in zip([prediction, ground_truth], ["pred", "truth"])]
            #
            # create_mr_big_img(transformed_image,
            #                   save_loc + "dwi_tran.png",
            #                   define_dvalues_big(transformed_image),
            #                   'png',
            #                   dpi=300)


            results.loc[results.id == name, 'size_ml'] = size_ml
            results.loc[results.id == name, 'size_pred_ml'] = size_pred_ml
            results.loc[results.id == name, 'dice'] = dice_score

        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

    print(f"Mean dice on test set: {metric:.4f}")

    results['mean_dice'] = metric

    # from sklearn.cluster import k_means
    # kmeans_labels = k_means(
    #     np.reshape(np.asarray(results['size'].to_list()), (-1,1)),
    #     n_clusters=2,
    #     random_state=0)[1]

    print(results)
    results.to_csv(root_dir + 'out_' + out_tag + '/results.csv', index=False)


if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    main()
