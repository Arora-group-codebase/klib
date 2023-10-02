import torch
from typing import List
import numpy as np
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms, datasets

from klib.kdataloader import KDataLoaderFFCV, KDataLoaderTorch, KDataLoader

from klib import train_utils

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

RESIZE_SIZE = 256
OUTPUT_SIZE = 224

DEFAULT_CROP_RATIO = OUTPUT_SIZE / RESIZE_SIZE

__all__ = ['get_kdataloader']


def get_torch_dataloader(
    data_pth, *, batch_size, num_workers, drop_last, device, train, seed,
    shuffle, replacement, distributed
) -> KDataLoaderTorch:

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if train:
        dataset = datasets.ImageFolder(
            data_pth,
            transforms.Compose([
                transforms.RandomResizedCrop(OUTPUT_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )
        return KDataLoaderTorch(
            train_utils.get_torch_dataloader_for_dataset(
                dataset, batch_size=batch_size, num_workers=num_workers,
                drop_last=drop_last, seed=seed, shuffle=shuffle, replacement=replacement, distributed=distributed
            ), device
        )
    else:
        dataset = datasets.ImageFolder(
            data_pth,
            transforms.Compose([
                transforms.Resize(RESIZE_SIZE),
                transforms.CenterCrop(OUTPUT_SIZE),
                transforms.ToTensor(),
                normalize,
            ])
        )
        return KDataLoaderTorch(
            train_utils.get_torch_dataloader_for_dataset(
                dataset, batch_size=batch_size, num_workers=num_workers,
                drop_last=False, seed=seed, shuffle=shuffle, replacement=replacement, distributed=distributed
            ), device
        )

        
def get_ffcv_dataloader(
    data_pth, *, batch_size, num_workers, drop_last, device, train, seed,
    shuffle, replacement, distributed
) -> KDataLoaderFFCV:

    assert not replacement

    if train:
        assert shuffle

        decoder = RandomResizedCropRGBImageDecoder((OUTPUT_SIZE, OUTPUT_SIZE))
        image_pipeline: List[Operation] = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN * 255, IMAGENET_STD * 255, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(device, non_blocking=True)
        ]

        loader = Loader(data_pth,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.RANDOM,
                        os_cache=True,
                        drop_last=drop_last,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed,
                        seed=seed,
                        batches_ahead=3)
    else:
        assert not shuffle
        
        cropper = CenterCropRGBImageDecoder((OUTPUT_SIZE, OUTPUT_SIZE), ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN * 255, IMAGENET_STD * 255, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(device, non_blocking=True)
        ]

        loader = Loader(data_pth,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed,
                        seed=seed,
                        batches_ahead=3)
    
    return KDataLoaderFFCV(loader)


def get_kdataloader(
    backend, data_pth, **kwargs
) -> KDataLoader:
    
    if backend == 'ffcv':
        return get_ffcv_dataloader(data_pth, **kwargs)
    else:
        assert backend == 'torch'
        return get_torch_dataloader(data_pth, **kwargs)