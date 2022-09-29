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
from monai.data import CacheDataset, DataLoader
from basic_model import SimpleSegmentationModel
from monai.transforms import (
    Compose,
    LoadImaged,
    CropForegroundd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    EnsureChannelFirstd,
    EnsureTyped,
    Resized
)
from monai.utils import first, set_determinism


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
    print(rank.type)
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    directory = '/media/mbcneuro/HDD1/DWI_Training_Data/'
    root_dir = directory

    train_files = make_dict(root_dir, "train")[:8]
    set_determinism(seed=42)

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

    # here we don't cache any data in case out of memory issue

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=4
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=4)

    model = SimpleSegmentationModel(2, 1).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    step = 0
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
    # update parameters
        optimizer.step()
    print("Training Loss: {}".format(loss.item()))


def main():
    world_size = 2
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
