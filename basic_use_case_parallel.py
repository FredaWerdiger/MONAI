import monai.networks.nets
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchmetrics import Dice
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import glob
import random
import matplotlib.pyplot as plt
from basic_model import SimpleSegmentationModel
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import (
    AsDiscrete,
    Compose,
    LoadImaged,
    CropForegroundd,
    EnsureChannelFirst,
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

#https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51

def cleanup():
    dist.destroy_process_group()


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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '2'
    os.environ['LOCAL RANK'] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def example(rank, world_size):
    print(f"Running DDP on rank {rank}.")
    # create default process group
    setup(rank, world_size)
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
    train_files = train_files
    val_files = val_files

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
                num_samples=2,
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
    batch_size = 2
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=4
    )

    train_loader = prepare(train_ds, rank, world_size, batch_size)
    # train_loader = DataLoader(train_ds,
    #                           batch_size=batch_size,
    #                           num_workers=0,
    #                           drop_last=False,
    #                           shuffle=False)

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=4
    )
    # val_loader = DataLoader(val_ds,
    #                           batch_size=batch_size,
    #                           num_workers=0,
    #                           drop_last=False,
    #                           shuffle=False)
    val_loader = prepare(val_ds, rank, world_size, batch_size)

    set_determinism(seed=42)
    # plot validation data example
    # l = random.randint(0, len(val_ds) - 1)
    # random_image = val_ds[l]
    # image_shape = random_image["image"].shape
    # channel_1 = random_image["image"][0]
    # channel_2 = random_image["image"][1]
    # label = random_image["label"][0]
    # random_slice = random.randint(0, image_shape[2] - 1)
    #
    # plt.figure("image",(18,6))
    # plt.subplot(1,3,1)
    # plt.imshow(channel_1[:, :, random_slice], cmap="gray")
    # plt.axis("off")
    # plt.title("Diffusion Weighted Imaging")
    # plt.subplot(1,3,2)
    # plt.imshow(channel_2[:, :, random_slice], cmap="gray")
    # plt.axis("off")
    # plt.title("Apparent Diffusion Coefficient")
    # plt.subplot(1,3,3)
    # plt.imshow(label[:, :, random_slice], cmap="gray")
    # plt.axis("off")
    # plt.title("Stroke lesion")
    # plt.show()

    model = SimpleSegmentationModel(2, 2).to(rank)
    model = monai.networks.nets.UNet(spatial_dims=3,
                                     in_channels=2,
                                     out_channels=2,
                                     channels=[32, 64, 128],
                                     strides=[2,2],
                                     kernel_size=3,
                                     num_res_units=2).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = monai.losses.DiceLoss(
        smooth_nr=0,
        smooth_dr=1e-5,
        to_onehot_y=True,
        softmax=True)
    optimizer = optim.Adam(
        ddp_model.parameters(),
        lr=1e-4,
        weight_decay=1e-5)
    # metric = DiceMetric(include_background=False)
    # metric to aggregate over ddt
    metric = Dice(dist_sync_on_step=True, ignore_index=0).to(rank)
    num_epochs = 300
    val_interval = 2
    num_batches = len(train_ds) / batch_size
    # only doing eval on master node
    if rank == 0:
        epoch_loss_list = [] # make a list of average loss values for the loss plot
        mean_dice_list = [] # make a list of mean dice loss
        best_metric = -1
        best_metric_epoch = -1
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True,to_onehot=None)])
    post_pred_label = Compose([EnsureType(), AsDiscrete(argmax=False, to_onehot=None)])
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        step = 0 # which step out of the number of batches
        epoch_loss = 0 # total loss for this epoch
        model.train()
        # train_loader.sampler.set_epoch(epoch)
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
        if rank == 0:
            epoch_loss /= step
            epoch_loss_list.append(epoch_loss)
        print(f"Average loss for epoch {epoch}: {epoch_loss:.4f}")

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
                    # val_outputs_list = post_pred(val_outputs) #for i in decollate_batch(val_outputs)]
                    # val_labels = post_pred_label(val_labels)# for i in decollate_batch(val_labels)]
                    # print(val_outputs_list.shape, val_outputs[0].max())
                    metric(val_outputs, val_labels.long())
                mean_dice = metric.compute().cpu().detach().numpy()
                metric.reset()
                if rank == 0:
                    mean_dice_list.append(mean_dice)
                    if mean_dice > best_metric:
                        CHECKPOINT_PATH = root_dir + 'practice/best_model.pth'
                        best_metric_epoch = epoch + 1
                        # only save in one process
                        if rank == 0:
                            torch.save(model.state_dict(), CHECKPOINT_PATH)
                        best_metric = mean_dice
                    print(
                        f"Mean dice at epoch {epoch + 1}: {mean_dice:.4f}"
                        f"\nBest mean dice: {best_metric:.4f}; at epoch {best_metric_epoch}"
                    )
    # now plot the loss and the dice
    if rank == 0:
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
        plt.savefig(root_dir + "practice/loss_plot.png", bbox_inches='tight', dpi=300, format='png')
        plt.close()
    cleanup()


def main():
    # comment out below for dev
    world_size = 2
    mp.spawn(example,
             args=(world_size,),
             nprocs=world_size,
             join=True)
if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    main()
