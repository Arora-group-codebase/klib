# klib
Kaifeng's library for training neural nets

# How to setup

Install conda

```bash
conda create -y -n klib pkg-config
conda activate klib
<<<<<<< HEAD
conda install libjpeg-turbo -c conda-forge
conda install pytorch=*=*cuda* torchvision pytorch-cuda=11.7 opencv -c pytorch -c nvidia
conda install numba
pip install cupy-cuda11x
conda update ffmpeg
pip install -r requirements.txt
```

# Compilers?

If the installation fails, check if your compilers are too old. If so, do this:

```bash
conda install -y compilers -c conda-forge
```

If some error occurs when you run the code, try the following.
Write the following content to `<miniconda-path>/envs/klib/etc/conda/activate.d/env_vars.sh`
```bash
#!/bin/sh
  
export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
```

Write the following content to `<miniconda-path>/envs/klib/etc/conda/deactivate.d/env_vars.sh`
```bash
#!/bin/sh

export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}
unset OLD_LD_LIBRARY_PATH
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

