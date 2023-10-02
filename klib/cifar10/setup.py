import klib.trainer
from .data import get_kdataloader
from klib.metric import count_correct
import argparse

from . import arch

__all__ = ['CIFAR10TrainerSetup']


VAL_B = 1000


class CIFAR10TrainerSetup(klib.trainer.BaseTrainerSetup):

    @classmethod
    def init_data(cls, self):
        kwargs = dict(
            data_pth=self.args.data_pth,
            backend=self.args.dataloader_backend,
            num_workers=self.args.n_dataloader_workers,
            seed=self.args.seed,
            device=self.device,
            distributed=self.world_size > 1
        )
        
        self.train_dataloader = get_kdataloader(
            batch_size=self.local_batch_size, drop_last=True, train=1, 
            crop=self.args.aug_crop, hflip=self.args.aug_hflip,
            shuffle=True, replacement=self.args.sample_with_replacement,
            subset = self.args.subset, subset_size=self.args.subset_size, num_classes = self.args.num_classes,
            **kwargs
        )

        self.test_dataloader = {
            'test': get_kdataloader(
                batch_size=VAL_B, drop_last=False, train=0,
                crop=False, hflip=False,
                shuffle=False, replacement=False,
                **kwargs
            )
        }
        if self.args.test_on_train:
            self.test_dataloader.update({
                'test_on_train': get_kdataloader(
                    batch_size=VAL_B, drop_last=False, train=1,
                    crop=False, hflip=False,
                    shuffle=False, replacement=False,
                    **kwargs
                )
            })
    
    
    @classmethod
    def init_extra_data(cls, self):
        if self.kmodel.has_bn():
            self.bn_dataloader = get_kdataloader(
                backend=self.args.dataloader_backend,
                data_pth=self.args.data_pth, batch_size=self.bn_batch_size,
                num_workers=self.args.n_dataloader_workers, drop_last=True, device=self.device, train=1, seed=self.args.seed,
                crop=self.args.aug_crop, hflip=self.args.aug_hflip,
                shuffle=True, replacement=self.args.sample_with_replacement,
                distributed=self.world_size > 1, subset = self.args.subset, subset_size=self.args.subset_size, num_classes = self.args.num_classes
            )
        else:
            self.bn_dataloader = None
    
    
    @classmethod
    def init_get_metrics(cls, self):
        def metric_acc(output, target):
            return count_correct(output, target, topk=(1,))
        
        return [metric_acc], ["acc"]
        

    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser):
        super().add_argparse_args(parser)
        parser.set_defaults(**{
            'lr_sche_type': 'multistep',
            'base_batch_size_for_lr': 128,
            'arch_lib': ['klib.cifar10'],
        })

        group = parser.add_argument_group('CIFAR-10 special')
        group.add_argument('--dataloader-backend', type=str, default='ffcv')
        group.add_argument('--n-dataloader-workers', type=int, default=4)
        group.add_argument('--num-classes', type=int, default=10)
        group.add_argument('--criterion', type=str, default='ce')
        group.add_argument('--test-on-train', type=int, default=1)
        group.add_argument('--aug-crop', type=int)
        group.add_argument('--aug-hflip', type=int)
        group.add_argument('--sample-with-replacement', type=int, default=0)
        group.add_argument('--subset', type=int, default=0)
        group.add_argument('--subset-size', type=int, default=0)

        

    @classmethod
    def input_size(cls, self):
        return (3, 32, 32)
