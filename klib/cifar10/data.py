from typing import List

import torch
import torchvision

from torchvision import transforms, datasets
import torch.utils.data

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

from klib.kdataloader import KDataLoaderFFCV, KDataLoaderTorch, KDataLoader

from klib import train_utils

import os
import os.path

import numpy as np


CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010])


def get_torch_dataloader(
    data_pth, *, batch_size, num_workers, drop_last, device, train, seed,
    crop, hflip, replacement=False, distributed=False
) -> KDataLoaderTorch:

    transform_list = []
    if crop:
        transform_list.append(transforms.RandomCrop(32, padding=4))
    if hflip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    
    dataset = datasets.CIFAR10(
        root=data_pth,
        train=train,
        transform=transforms.Compose(transform_list)
    )

    return KDataLoaderTorch(
        train_utils.get_torch_dataloader_for_dataset(
            dataset, batch_size=batch_size, num_workers=num_workers,
            drop_last=drop_last, seed=seed, shuffle=train, replacement=replacement, distributed=distributed
        ), device
    )


def get_ffcv_dataloader(
    data_pth, *, batch_size, num_workers, drop_last, device, train, seed,
    crop, hflip, replacement=False, distributed=False
) -> KDataLoaderFFCV:

    assert not replacement

    if train:
        ffcv_fname = '/tmp/cifar10_ffcv/train.beton'
    else:
        ffcv_fname = '/tmp/cifar10_ffcv/test.beton'

    if not os.path.isfile(ffcv_fname):
        os.makedirs('/tmp/cifar10_ffcv', exist_ok=True)

        dataset = datasets.CIFAR10(root=data_pth, train=train)
        fields = {
            'image': RGBImageField(),
            'label': IntField(),
        }
        writer = DatasetWriter(ffcv_fname, fields)
        writer.from_indexed_dataset(dataset)
    
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
    image_pipeline = [SimpleRGBImageDecoder()]

    if crop:
        image_pipeline.extend([
            RandomTranslate(padding=2),
            Cutout(8, tuple(map(int, CIFAR_MEAN * 255))), # Note Cutout is done before normalization.
        ])
    
    if hflip:
        image_pipeline.append(
            RandomHorizontalFlip(),
        )
    
    image_pipeline.extend([
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        Convert(torch.get_default_dtype()),
        transforms.Normalize(CIFAR_MEAN * 255, CIFAR_STD * 255),
    ])

    return KDataLoaderFFCV(
        Loader(
            ffcv_fname,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.RANDOM if train else OrderOption.SEQUENTIAL,
            drop_last=drop_last,
            pipelines={
                'image': image_pipeline,
                'label': label_pipeline
            },
            distributed=distributed,
            seed=seed
        )
    )


def get_kdataloader(
    backend, data_pth, **kwargs
) -> KDataLoader:
    
    if backend == 'ffcv':
        return get_ffcv_dataloader(data_pth, **kwargs)
    else:
        assert backend == 'torch'
        return get_torch_dataloader(data_pth, **kwargs)
