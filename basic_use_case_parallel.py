import monai.networks.nets
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.functional import one_hot
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
import glob
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
from basic_model import SimpleSegmentationModel
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import (
    AsDiscrete,
    Compose,
    LoadImaged,
    CropForegroundd,
    EnsureType,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    EnsureChannelFirstd,
    EnsureTyped,
    Resized
)

from monai.inferers import sliding_window_inference
from monai.utils import first, set_determinism
from monai.metrics import DiceMetric


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


def example(rank, world_size):
    # create default process group
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    if os.path.exists('/media/'):
        directory = '/media/mbcneuro/HDD1/DWI_Training_Data/'
    else:
        directory = 'D:/ctp_project_data/DWI_Training_Data/'

    root_dir = directory
    if not os.path.exists(root_dir + "practice"):
        os.makedirs(root_dir + "practice")


    train_files, val_files = [make_dict(root_dir, string) for string in ["train", "validation"]]

    # reduce dataset for now
    train_files = train_files[:16]
    val_files = val_files[:8]

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
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64, 64, 64),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Resized(keys=["image", "label"],
                    mode=['trilinear', "nearest"],
                    align_corners=[True, None],
                    spatial_size=(128, 128, 128)),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    # here we don't cache any data in case out of memory issue
    batch_size = 4
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=4
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=4
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    set_determinism(seed=42)

    import matplotlib.pyplot as plt
    # plot validation data example
    l = random.randint(0, len(val_ds) - 1)
    random_image = val_ds[l]
    image_shape = random_image["image"].shape
    channel_1 = random_image["image"][0]
    channel_2 = random_image["image"][1]
    label = random_image["label"][0]
    random_slice = random.randint(0, image_shape[2] - 1)

    plt.figure("image",(18,6))
    plt.subplot(1,3,1)
    plt.imshow(channel_1[:, :, random_slice], cmap="gray")
    plt.axis("off")
    plt.title("Diffusion Weighted Imaging")
    plt.subplot(1,3,2)
    plt.imshow(channel_2[:, :, random_slice], cmap="gray")
    plt.axis("off")
    plt.title("Apparent Diffusion Coefficient")
    plt.subplot(1,3,3)
    plt.imshow(label[:, :, random_slice], cmap="gray")
    plt.axis("off")
    plt.title("Stroke lesion")
    plt.show()

    model = SimpleSegmentationModel(2, 2).to(rank)
    # model = monai.networks.nets.UNet(spatial_dims=3,
    #                                  in_channels=2,
    #                                  out_channels=2,
    #                                  channels=[32, 64, 128],
    #                                  strides=[2,2],
    #                                  kernel_size=3,
    #                                  num_res_units=2).to(rank)
    # construct DDP model
    if rank=="cuda":
        ddp_model = model
    else:
        ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    metric = DiceMetric(include_background=False)
    num_epochs = 20
    val_interval = 2
    num_batches = len(train_ds) / batch_size
    epoch_loss_list = [] # make a list of average loss values for the loss plot
    mean_dice_list = [] # make a list of mean dice loss
    best_metric = -1
    best_metric_epoch = -1
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True,to_onehot=2)])
    post_pred_label = Compose([EnsureType(), AsDiscrete(argmax=False, to_onehot=2)])
    for epoch in tqdm(range(num_epochs)):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        step = 0 # which step out of the number of batches
        epoch_loss = 0 # total loss for this epoch
        model.train()
        for batch in train_loader:
            step += 1
            # forward pass
            inputs, labels = (
                batch["image"],
                batch["label"]
            )
            # inputs = torch.randn((1, 2, 128, 128, 128))
            # labels = torch.randn((1, 2, 128, 128, 128))
            outputs = ddp_model(inputs.to(rank))
            labels = labels.to(rank)
            # backward pass
            loss = loss_fn(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            # update parameters
            optimizer.step()
            print("step {}/{}".format(step, int(num_batches)))
            print(f"Training Loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_list.append(epoch_loss)
        print(f"Average loss for epoch: {epoch_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            model.eval()
            print("Evaluating...")
            with torch.no_grad():
                for val in val_loader:
                    val_inputs, val_labels = (
                        val["image"].to(rank),
                        val["label"].to(rank)
                    )
                    roi_size = (64, 64, 64)
                    sw_batch_size = 2
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    # transform from a batched tensor to a list of tensors
                    # turn into an array of discrete binary values
                    val_outputs_list = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_pred_label(i) for i in decollate_batch(val_labels)]
                    metric(y_pred=val_outputs_list, y=val_labels)
                mean_dice = metric.aggregate().item()
                metric.reset()
                mean_dice_list.append(mean_dice)
                if mean_dice > best_metric:
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), root_dir + 'practice/best_model.pth')
                    best_metric = mean_dice
                print(
                    f"Mean dice at epoch {epoch + 1}: {mean_dice:.4f}"
                    f"\nBest mean dice: {best_metric:.4f}; at epoch {best_metric_epoch}"
                )
        sleep(0.02)

    # now plot the loss and the dice
    plt.figure("Results of training", (12,6))
    plt.subplot(1, 2, 1)
    x = [i + 1 for i in range(len(epoch_loss_list))]
    plt.plot(x, epoch_loss_list)
    plt.title("Loss trend")
    plt.xlabel("Epoch number")
    plt.ylabel("Average loss")
    plt.subplot(1, 2, 2)
    plt.title('Mean dice trend')
    plt.xlabel("Epoch number")
    plt.ylabel("Mean dice")
    x = [(i + 1) * val_interval for i in range(len(mean_dice_list))]
    plt.plot(x, mean_dice_list)
    plt.savefig(root_dir + "practice", bbox_inches='tight', dpi=300, format='png')
    plt.close()

def main():
    # comment out below for dev
    world_size = 1
    mp.spawn(example,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
