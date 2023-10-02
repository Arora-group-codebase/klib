import argparse
from pathlib import Path
import torch
import torchvision
import os
import wandb
import yaml

from klib.trainer import StochasticTrainer
from klib.cifar10 import CIFAR10TrainerSetup
from klib.train_utils import parse_trainer_args

config = yaml.load(open('.config.yml'), Loader=yaml.FullLoader)

def main(args):
    assert torch.cuda.is_available()

    trainer = StochasticTrainer(args, CIFAR10TrainerSetup)
    
    if trainer.rank == 0:
        os.makedirs(trainer.args.save_pth, exist_ok=True)
        init_wandb(trainer.args)
    
    trainer.run()


def init_wandb(args):
    wandb.init(
        mode=args.wandb,
        project="cifar-base",
        entity=config['wandb_entity'],
        name=f"{Path(args.recipe_pth).stem}-lr={args.lr}-model={args.arch}-m={args.beta1}",
        config=vars(args),
    )
    wandb.run.log_code(".")
    #settings=wandb.Settings(start_method='fork'),


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    CIFAR10TrainerSetup.add_argparse_args(parser)

    parser.set_defaults(**{
        'recipe_pth': 'recipe/cifar10-preresnet32-B128.yml',
    })

    # scale-invariance
    parser.add_argument('--final-layer-code', default='linear-fixed-etf')

    # data pth
    parser.add_argument('--data-pth', default=config['cifar10_data_pth'])

    parser.add_argument('--wandb', type=str, default=None)

    args = parse_trainer_args(parser)
    
    main(args)
