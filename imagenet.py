import argparse
from pathlib import Path
import torch
import torchvision
import os
import wandb
import yaml

from klib.trainer import StochasticTrainer
from klib.imagenet import ImageNetTrainerSetup

config = yaml.load(open('.config.yml'), Loader=yaml.FullLoader)

def main(args):
    assert torch.cuda.is_available()

    trainer = StochasticTrainer(args, ImageNetTrainerSetup)
    
    if trainer.rank == 0:
        os.makedirs(trainer.args.save_pth, exist_ok=True)
        init_wandb(trainer.args)
    
    trainer.run()


def init_wandb(args):
    wandb.init(
        project="imagenet-base",
        entity=config['wandb_entity'],
        name=f"{Path(args.recipe_pth).stem}-lr={args.lr}-m={args.beta1}-{args.autocast_dtype}",
        config=vars(args)
    )
    wandb.run.log_code(".")
    #settings=wandb.Settings(start_method='fork'),


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--debug', type=int, default=0)

    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    parser.add_argument('--recipe-pth', default='recipe/imagenet-resnet50-B2048.yml', help='path to the training recipe (an YAML file)')

    # data loader
    parser.add_argument('--dataloader-backend', type=str, default='ffcv')
    parser.add_argument('--n-dataloader-workers', type=int, default=6)

    # batch size
    parser.add_argument('--total-batch-size', type=int)
    parser.add_argument('--physical-batch-size', type=int)
    parser.add_argument('--bn-batch-size', type=int)

    # hyperparam
    parser.add_argument('--arch', type=str, help='model architecture')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--beta1', type=float, help='momentum value')
    parser.add_argument('--nesterov', type=int, help='whether to use nesterov momentum')
    parser.add_argument('--wd', type=float, help='weight decay value')

    # normalization
    parser.add_argument('--norm-layer', type=str, default='bn')
    parser.add_argument('--bn-eps', type=float, default=1e-5)
    parser.add_argument('--bn-momentum', type=float, default=0.1)
    parser.add_argument('--bn-batches', type=int, default=64)

    # scheduler
    parser.add_argument('--warmup', type=int, help='whether to use lr warmup')
    parser.add_argument('--warmup-epochs', type=int)
    parser.add_argument('--warmup-start-lr', type=float)
    parser.add_argument('--lrdecay-sche', nargs='+', type=int)
    parser.add_argument('--gamma', type=float, default=0.1, help='the factor for learning rate decay')

    # resume
    parser.add_argument('--resume', type=int, default=0, help='the epoch to continue training')
    parser.add_argument('--resume-pth', type=str, help='the path to load the model to resume')
    
    # save
    parser.add_argument('--save-freq', type=int, default=-1)
    parser.add_argument('--save-latest', type=int, default=0)

    # eval
    parser.add_argument('--eval-freq', type=int, default=1)

    # low-precision training
    parser.add_argument('--autocast-dtype', type=str)
    parser.add_argument('--grad-upscale', type=int)


    args = parser.parse_args()
    if args.dataloader_backend == 'ffcv':
        args.train_pth = config['imagenet_ffcv_train_pth']
        args.val_pth = config['imagenet_ffcv_val_pth']
    else:
        args.train_pth = config['imagenet_torch_train_pth']
        args.val_pth = config['imagenet_torch_val_pth']
    args.save_pth = './check_point/' + Path(args.recipe_pth).stem + f'-{args.seed}'
    
    main(args)
