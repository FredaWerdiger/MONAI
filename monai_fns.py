import os
import glob
import torch.distributed as dist
import torch
from monai.data import Dataset, DataLoader, DistributedSampler
from numba import cuda
from GPUtil import showUtilization as gpu_usage


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


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '2'
    os.environ['LOCAL RANK'] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class DDPSetUp():
    def __init__(self,
                rank,
                world_size,
                ):
        self.cleanup = dist.destroy_process_group()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = '2'
        os.environ['LOCAL RANK'] = str(rank)
        self.setup = dist.init_process_group("nccl", rank=rank, world_size=world_size)


class BuildDataset():
    def __init__(self, directory, string):
        images = sorted(
            glob.glob(os.path.join(directory, string, 'images', '*.nii.gz'))
        )
        labels = sorted(
            glob.glob(os.path.join(directory, string, 'masks', '*.nii.gz'))
        )
        self.images_dict = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(images, labels)
        ]
        self.no_seg_dict = [
            {"image": image_name} for image_name in images
                            ]


