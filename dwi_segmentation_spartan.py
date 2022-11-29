# following tutorial from BRATs segmentation
# two classes insead of 4 classes
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import math
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
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
    SaveImaged
)

from monai.utils import first, set_determinism

import torch
import os
from recursive_data import get_semi_dataset




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
    existing_model = directory + 'out_final_no_cropping/best_metric_model600.pth'

    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)

    # create outdir
    out_tag = "unet_recursive"
    if not os.path.exists(root_dir + 'out_' + out_tag):
        os.makedirs(root_dir + 'out_' + out_tag)

    train_files, val_files, test_files = [
        make_dict(root_dir, string) for string in ['train', 'validation', 'test']]
    semi_files = get_semi_dataset()
    train_files = semi_files + train_files

    set_determinism(seed=42)

    max_epochs = 600
    batch_size = 2
    image_size = (128, 128, 128)
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
            # RandAdjustContrastd(keys="image", prob=1, gamma=(0.5, 1)),
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

    # here we don't cache any data in case out of memory issue
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=4
    )

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True)

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=4)

    val_loader = DataLoader(val_ds,
                            batch_size=batch_size)

    rank = 'cuda'

    # Uncomment to display data
    #
    # import random
    # m = random.randint(0, 50)
    # s = random.randint(0, 63)
    # val_data_example = val_ds[m]
    # print(f"image shape: {val_data_example['image'].shape}")
    # plt.figure("image", (18, 6))
    # for i in range(2):
    #     plt.subplot(1, 3, i + 1)
    #     plt.title(f"image channel {i}")
    #     plt.imshow(val_data_example["image"][i, :, :, s].detach().cpu(), cmap="gray")
    # # also visualize the 3 channels label corresponding to this image
    # print(f"label shape: {val_data_example['label'].shape}")
    # plt.subplot(1, 3, 3)
    # plt.title("label")
    # plt.imshow(val_data_example["label"][0, :, :, s].detach().cpu())
    # plt.show()
    # plt.close()

    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(rank)
    # model = AttentionUnet(
    #     spatial_dims=3,
    #     in_channels=2,
    #     out_channels=2,
    #     channels=(32, 64, 128, 256, 512),
    #     strides=(2, 2, 2, 2),
    # ).to(rank)

    # model = DenseNet(
    #     spatial_dims=3,
    #     in_channels=2,
    #     out_channels=2
    # ).to(rank)

    # model = SegResNet(
    #     blocks_down=[1, 2, 2, 4],
    #     blocks_up=[1, 1, 1],
    #     init_filters=8,
    #     in_channels=2,
    #     out_channels=2,
    #     dropout_prob=0.2,
    # ).to(device)

    loss_function = DiceLoss(
        smooth_nr=0,
        smooth_dr=1e-5,
        to_onehot_y=True,
        softmax=True,
        include_background=False)
    optimizer = torch.optim.Adam(
        model.parameters(),
        1e-4,
        weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

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
    # load existing model
    model.load_state_dict(torch.load(existing_model))
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        epoch_loss = 0
        step = 0
        model.train()
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(rank),
                batch_data["label"].to(rank),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            # commenting out print function
            # print(
            #     f"{step}/{len(train_ds) // train_loader.batch_size}, "
            #     f"train_loss: {loss.item():.4f}")
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
                        val_data["image"].to(rank),
                        val_data["label"].to(rank),
                    )
                    # unsure how to optimize this
                    roi_size = (64, 64, 64)
                    sw_batch_size = 2
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
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
    time_taken_hours = time_taken/3600
    time_taken_mins = np.ceil((time_taken/3600 - int(time_taken/3600)) * 60)
    time_taken_hours = int(time_taken_hours)

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
    plt.close()
    # compare dice with f1
    # plt.figure("Compare dice scores", (6, 6))
    # plt.title("Compare dice scores")
    # x = [val_interval * (i + 1) for i in range(len(metric_values))]
    # y = metric_values
    # plt.plot(x, y, 'b', label="synchronized mean dice")
    # y = f1_mean_values
    # plt.plot(x, y, 'k', label="manual mean dice")
    # plt.legend()
    # plt.savefig(os.path.join(root_dir + 'out_' + out_tag, model_name.split('.')[0] + 'dice_compare.png'),
    #             bbox_inches='tight', dpi=300, format='png')
    # plt.close()


    # save model results in a separate file
    with open(root_dir + 'out_' + out_tag + '/model_info.txt', 'w') as myfile:
        myfile.write(f'Number of epochs: {max_epochs}\n')
        myfile.write(f'Batch size: {batch_size}\n')
        myfile.write(f'Image size: {image_size}\n')
        myfile.write(f'Validation interval: {val_interval}\n')
        myfile.write(f"Best metric: {best_metric:.4f}\n")
        myfile.write(f"Best metric epoch: {best_metric_epoch}\n")
        myfile.write(f"Time taken: {time_taken_hours} hours, {time_taken_mins} mins\n")

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

    '''
    Below will be shifted to another code as this part does not need to be distributed
    '''
    # # test on external data
    # test_transforms = Compose(
    #     [
    #         LoadImaged(keys=["image", "label"]),
    #         EnsureChannelFirstd(keys="image"),
    #         Resized(keys="image",
    #                 mode='trilinear',
    #                 align_corners=True,
    #                 spatial_size=(128, 128, 128)),
    #         NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    #         EnsureTyped(keys=["image", "label"]),
    #         SaveImaged(keys="image", output_dir=root_dir + "out", output_postfix="transform", resample=False)
    #     ]
    # )
    #
    #
    # test_ds = Dataset(
    #     data=test_files, transform=test_transforms)
    #
    # test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)
    #
    # post_transforms = Compose([
    #     EnsureTyped(keys=["pred", "label"]),
    #     EnsureChannelFirstd(keys="label"),
    #     Invertd(
    #         keys="pred",
    #         transform=test_transforms,
    #         orig_keys="image",
    #         meta_keys="pred_meta_dict",
    #         orig_meta_keys="image_meta_dict",
    #         meta_key_postfix="meta_dict",
    #         nearest_interp=False,
    #         to_tensor=True,
    #     ),
    #     AsDiscreted(keys="pred", argmax=True, to_onehot=2),
    #     AsDiscreted(keys="label", to_onehot=2),
    #     SaveImaged(keys="pred",
    #                meta_keys="pred_meta_dict",
    #                output_dir=root_dir + "out_" + out_tag,
    #                output_postfix="seg", resample=False),
    # ])
    #
    # from monai.transforms import LoadImage
    #
    # if rank == 0:
    #     # removing sync on step as we are running on master node
    #     dice_metric = Dice()
    #     loader = LoadImage(image_only=False)
    #     model.load_state_dict(torch.load(
    #         os.path.join(root_dir, 'out_' + out_tag, model_name)))
    #
    #     model.eval()
    #
    #     results = pd.DataFrame(columns=['id', 'dice', 'size', 'px_x', 'px_y', 'py_z', 'size_ml'])
    #     results['id'] = ['test_' + str(item).zfill(3) for item in range(1, len(test_loader) + 1)]
    #
    #     with torch.no_grad():
    #         for i, test_data in enumerate(test_loader):
    #             test_inputs = test_data["image"].to(rank)
    #
    #             roi_size = (64, 64, 64)
    #             sw_batch_size = 2
    #             test_data["pred"] = sliding_window_inference(
    #                 test_inputs, roi_size, sw_batch_size, model)
    #
    #             test_data = [post_transforms(i) for i in decollate_batch(test_data)]
    #
    #             test_output, test_label, test_image = from_engine(["pred", "label", "image"])(test_data)
    #
    #             a = dice_metric(test_output[0], test_label[0].long())
    #
    #             dice_score = round(a.item(), 4)
    #
    #             # get original image, and normalize it so we can see the normalized image
    #             # this is both channels
    #             original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])
    #             volx, voly, volz = original_image[1]['pixdim'][1:4] # meta data
    #             pixel_vol = volx * voly * volz
    #
    #             original_image = original_image[0] # image data
    #             original_adc = original_image[:, :, :, 1]
    #             original_image = original_image[:, :, :, 0]
    #             ground_truth = test_label[0][1].detach().numpy()
    #             prediction = test_output[0][1].detach().numpy()
    #             transformed_image = test_inputs[0][0].detach().cpu().numpy()
    #             size = prediction.sum()
    #             size_ml = size * pixel_vol / 1000
    #             name = "test_" + os.path.basename(
    #                 test_data[0]["image_meta_dict"]["filename_or_obj"]).split('.nii.gz')[0].split('_')[1]
    #             save_loc = root_dir + "out_" + out_tag + "/images/" + name + "_"
    #
    #             if not os.path.exists(root_dir + "out_" + out_tag + "/images/"):
    #                 os.makedirs(root_dir + "out_" + out_tag + "/images/")
    #
    #             create_paper_img(
    #                 original_image,
    #                 ground_truth,
    #                 prediction,
    #                 save_loc + "paper.png",
    #                 define_dvalues(original_image),
    #                 'png',
    #                 dpi=300
    #             )
    #             create_mr_img(
    #                 original_image,
    #                 save_loc + "dwi.png",
    #                 define_dvalues(original_image),
    #                 'png',
    #                 dpi=300)
    #             create_adc_img(
    #                 original_adc,
    #                 save_loc + "adc.png",
    #                 define_dvalues(original_image),
    #                 'png',
    #                 dpi=300)
    #
    #             [create_mrlesion_img(
    #                 original_image,
    #                 im,
    #                 save_loc + name + '.png',
    #                 define_dvalues(original_image),
    #                 'png',
    #                 dpi=300) for im, name in zip([prediction, ground_truth], ["pred", "truth"])]
    #
    #             create_mr_big_img(transformed_image,
    #                               save_loc + "dwi_tran.png",
    #                               define_dvalues_big(transformed_image),
    #                               'png',
    #                               dpi=300)
    #
    #             # uncomment below to visualise results.
    #             # plt.figure("check", (24, 6))
    #             # plt.subplot(1, 4, 1)
    #             # plt.imshow(original_image[:, :, 12], cmap="gray")
    #             # plt.title(f"image {name}")
    #             # plt.subplot(1, 4, 2)
    #             # plt.imshow(test_image[0].detach().cpu()[0, :, :, 12], cmap="gray")
    #             # plt.title(f"transformed image {name}")
    #             # plt.subplot(1, 4, 3)
    #             # plt.imshow(test_label[0].detach().cpu()[:, :, 12])
    #             # plt.title(f"label {name}")
    #             # plt.subplot(1, 4, 4)
    #             # plt.imshow(test_output[0].detach().cpu()[1, :, :, 12])
    #             # plt.title(f"Dice score {dice_score}")
    #             # plt.show()
    #
    #             results.loc[results.id == name, 'size'] = size
    #             results.loc[results.id == name, 'size_ml'] = size_ml
    #             results.loc[results.id == name, 'px_x'] = volx
    #             results.loc[results.id == name, 'px_y'] = voly
    #             results.loc[results.id == name, 'px_z'] = volz
    #             results.loc[results.id == name, 'dice'] = dice_score
    #
    #         # aggregate the final mean dice result
    #         metric = dice_metric.compute().cpu().detach().numpy()
    #         # reset the status for next validation round
    #         dice_metric.reset()
    #
    #     print(f"Mean dice on test set: {metric}")
    #
    #     results['mean_dice'] = metric
    #     try:
    #         results['training_hours'] = time_taken_hours
    #         results['training_minutes'] = time_taken_mins
    #     except NameError:
    #         print('No time taken.')
    #
    #     from sklearn.cluster import k_means
    #     kmeans_labels = k_means(
    #         np.reshape(np.asarray(results['size'].to_list()), (-1,1)),
    #         n_clusters=2,
    #         random_state=0)[1]
    #     kmeans_labels = ["small-medium" if label==0 else "medium-large" for label in kmeans_labels]
    #     results['size_label']=kmeans_labels
    #     results_join = results.join(
    #         ctp_df[~ctp_df.index.duplicated(keep='first')],
    #         on='id',
    #         how='left')
    #     print(results)
    #     results_join.to_csv(root_dir + 'out_' + out_tag + '/results.csv', index=False)
    #
    #     for sub in results_join['id']:
    #         create_overviewhtml(sub, results_join, root_dir + 'out_' + out_tag + '/')

if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    main()
