# klib
Kaifeng's library for training neural nets

# How to setup

```bash
conda create -y -n klib python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
conda activate klib
```

Some pip packages may be needed...

# How to train CIFAR-10

``` bash
./run-cifar.sh --recipe-pth recipe/cifar10-preresnet32-B128.yml
```

# How to train ImageNet

The following code uses 8 GPUs by default.

``` bash
./drun-imagenet.sh --recipe-pth recipe/imagenet-resnet50-B8192.yml
```

# Configs

## config.yml

`config.yml` can be used to configurate:

* The data paths to CIFAR-10/ImageNet
* wandb entity

## Command lines

You can pass arguments like `--lr 0.01`, `--wd 1e-5`, `--warmup 1 --warmup-epochs 5` to the command line.

## Recipes

Training recipes can be put under `recipe/`, providing default command line arguments.

