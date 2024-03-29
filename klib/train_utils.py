import torch
from torch import nn, optim
import torchvision
from klib.normalizer import GhostBatchNorm2d, GhostBatchNorm1d
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import importlib

import os
import yaml
import argparse

import klib.kmodel
import klib.metric
import klib.hooker
import klib.sche


TORCH_DTYPES = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
    'uint8': torch.uint8,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
}


def is_bn(m):
    return isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)


def is_normalizer(m):
    return is_bn(m) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm)


def get_flat_tensor_from_tensor_sequence(seq):
    all = []
    for p in seq:
        all.append(p.view(-1))
    return torch.cat(all)


def get_mean_flat_tensor_from_tensor_sequences(seqs):
    all = []
    for ps in zip(*seqs):
        all.append(torch.stack(ps).mean(dim=0).view(-1))
    return torch.cat(all)


def set_flat_tensor_to_tensor_sequence(flat, seq):
    idx = 0
    for p in seq:
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].view_as(p))
        idx += n


def load_builtin_model(args) -> nn.Module:
    for libname in args.arch_lib + ['klib']:
        lib = importlib.import_module(f'{libname}.arch')
        if args.arch in lib.__dict__:
            return lib.__dict__[args.arch](args)
    if args.arch in torchvision.models.__dict__:
        kwargs = dict(norm_layer=lambda w: get_norm2d(w, args), num_classes=args.num_classes)
        return klib.kmodel.KModel, torchvision.models.__dict__[args.arch](**kwargs)
    else:
        return None


def load_builtin_optimizer(model: nn.Module, args) -> optim.Optimizer:
    opt = args.opt.lower()
    
    no_wd_params = set()
    if args.no_wd_on_bias:
        for name, p in model.named_parameters():
            if name.endswith('.bias'):
                no_wd_params.add(p)
        
    if args.no_wd_on_normalizer:
        for m in model.modules():
            if is_bn(m):
                if m.weight is not None:
                    no_wd_params.add(m.weight)
                if m.bias is not None:
                    no_wd_params.add(m.bias)
    
    params = list(model.parameters())
    if no_wd_params:
        params = [{
            "params": list(set(params) - no_wd_params)
        }, {
            "params": list(no_wd_params),
            "weight_decay": 0
        }]
    
    if opt == 'sgd':
        return torch.optim.SGD(params, lr=args.lr, momentum=args.beta1, weight_decay=args.wd, nesterov=args.nesterov)
    elif opt == 'adamw':
        return torch.optim.AdamW(params, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.wd)
    elif opt.startswith('ext.'):
        name = opt[4:]
        return importlib.import_module(f'ext.optim.{name}').__dict__[name](params, args)
    else:
        return None


def load_builtin_criterion(args) -> nn.Module:
    if args.criterion == 'ce':
        return nn.CrossEntropyLoss()
    elif args.criterion == 'celnr':
        return klib.metric.CELNR(args.label_smoothing)
    else:
        assert args.criterion == 'bce'
        return nn.BCEWithLogitsLoss()


def load_builtin_hooker(name: str) -> klib.hooker.Hooker:
    if name.startswith('ext.'):
        name = name[4:]
        lib = 'ext'
    else:
        lib = 'klib'
    clsname = ''.join(map(lambda s: s.title(), name.split('_'))) + 'Hooker'
    return importlib.import_module(f'{lib}.hooker.{name}').__dict__[clsname]


def load_builtin_scheduler(name: str) -> klib.sche.KScheduler:
    if name.startswith('ext.'):
        name = name[4:]
        lib = 'ext'
    else:
        lib = 'klib'
    clsname = 'K' + ''.join(map(lambda s: s.title(), name.split('_'))) + 'Scheduler'
    return importlib.import_module(f'{lib}.sche.{name}').__dict__[clsname]


def get_torch_dataloader_for_dataset(
    dataset, *, batch_size, num_workers, drop_last, seed, shuffle=None, replacement=False, distributed=False, subset=False, subset_size=50000, num_classes=10):

    replacement = bool(replacement)

    if subset:
        indices = get_subset_indices(dataset, num_samples=subset_size, num_classes=num_classes)
        dataset = torch.utils.data.Subset(dataset, indices)
    
    print(f"dataset size: {len(dataset)}")



    if distributed and not replacement:
        sampler = torch.utils.data.DistributedSampler(dataset, seed=seed, drop_last=drop_last, shuffle=shuffle)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            num_workers=num_workers, pin_memory=True, sampler=sampler
        )
    
    assert not (not shuffle and replacement)
        
    if shuffle or replacement:
        gen = torch.Generator()
        if distributed:
            gen.manual_seed(seed + int(os.environ["RANK"]))
        else:
            gen.manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(dataset, replacement=replacement, generator=gen)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, sampler=sampler,
        drop_last=drop_last
    )

# only tested for torchvision datasets
def get_subset_indices(dataset, num_samples=50000, num_classes=10):
    assert num_samples % num_classes == 0, "the number of samples should be divisible by the number of classes"
    num_per_class = num_samples // num_classes
    indices = []
    for i in range(num_classes):
        class_indices = list(np.where(np.array(dataset.targets) == i)[0])
        indices.extend(class_indices[:num_per_class])
    return indices


def get_batchnorm2d(w, *, batch_size=None, **kwargs):
    if batch_size is None:
        return nn.BatchNorm2d(w, **kwargs)
    else:
        return GhostBatchNorm2d(w, batch_size, **kwargs)


def get_batchnorm1d(w, *, batch_size=None, **kwargs):
    if batch_size is None:
        return nn.BatchNorm1d(w, **kwargs)
    else:
        return GhostBatchNorm1d(w, batch_size, **kwargs)


def get_norm1d(w, args, eps=None, momentum=None, **kwargs):
    assert args.norm_layer == 'bn'
    return get_batchnorm1d(
        w,
        batch_size=args.bn_batch_size if args.physical_batch_size != args.bn_batch_size else None,
        eps=args.bn_eps if eps is None else eps, momentum=args.bn_momentum if momentum is None else momentum,
        **kwargs
    )


def get_norm2d(w, args, eps=None, momentum=None, **kwargs):
    assert args.norm_layer == 'bn'
    return get_batchnorm2d(
        w,
        batch_size=args.bn_batch_size if args.physical_batch_size != args.bn_batch_size else None,
        eps=args.bn_eps if eps is None else eps, momentum=args.bn_momentum if momentum is None else momentum,
        **kwargs
    )


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    else:
        raise NotImplementedError()


def parse_trainer_args(parser: argparse.ArgumentParser):
    args, _ = parser.parse_known_args()
    recipe_args = argparse.Namespace(**yaml.load(open(args.recipe_pth), Loader=yaml.FullLoader))
    args, _ = parser.parse_known_args(namespace=recipe_args)
    for hooker_name in args.hooker:
        load_builtin_hooker(hooker_name).add_argparse_args(parser)
    for s in dir(args):
        if s.endswith('_sche_type'):
            load_builtin_scheduler(getattr(args, s)).add_argparse_args(parser, s[:-10])
    if args.opt.startswith('ext.'):
        name = args.opt[4:]
        importlib.import_module(f'ext.optim.{name}').__dict__[name + '_add_argparse_args'](parser)
    args = parser.parse_args(namespace=recipe_args)
    return args
