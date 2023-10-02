import argparse
from pathlib import Path
import torch
import torchvision
import os
import wandb
import yaml

from klib.trainer import StochasticTrainer
from klib.imagenet import ImageNetTrainerSetup

from klib.train_utils import parse_trainer_args

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
    ImageNetTrainerSetup.add_argparse_args(parser)

    parser.set_defaults(**{
        'recipe_pth': 'recipe/imagenet-resnet50-B2048.yml',
    })

    # scale-invariance
    parser.add_argument('--final-layer-code', default='linear-fixed')

    args, _ = parser.parse_known_args()
    if args.dataloader_backend == 'ffcv':
        parser.add_argument('--train-pth', default=config['imagenet_ffcv_train_pth'])
        parser.add_argument('--val-pth', default=config['imagenet_ffcv_val_pth'])
    else:
        parser.add_argument('--train-pth', default=config['imagenet_torch_train_pth'])
        parser.add_argument('--val-pth', default=config['imagenet_torch_val_pth'])

    args = parse_trainer_args(parser)
    
    main(args)
