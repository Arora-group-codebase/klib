import klib.trainer
from .data import get_kdataloader
from klib.metric import count_correct
import argparse


__all__ = ['ImageNetTrainerSetup']


VAL_B = 256


class ImageNetTrainerSetup(klib.trainer.BaseTrainerSetup):
    
    
    @classmethod
    def init_data(cls, self):
        self.train_dataloader = get_kdataloader(
            backend=self.args.dataloader_backend,
            data_pth=self.args.train_pth, batch_size=self.local_batch_size,
            num_workers=self.args.n_dataloader_workers, drop_last=True, device=self.device, train=1, seed=self.args.seed,
            shuffle=True, replacement=self.args.sample_with_replacement,
            distributed=self.world_size > 1
        )

        self.test_dataloader = get_kdataloader(
            backend=self.args.dataloader_backend,
            data_pth=self.args.val_pth, batch_size=VAL_B,
            num_workers=self.args.n_dataloader_workers, drop_last=False, device=self.device, train=0, seed=self.args.seed,
            shuffle=False, replacement=False,
            distributed=self.world_size > 1
        )
    
    
    @classmethod
    def init_extra_data(cls, self):
        if self.kmodel.has_bn():
            self.bn_dataloader = get_kdataloader(
                backend=self.args.dataloader_backend,
                data_pth=self.args.train_pth, batch_size=self.bn_batch_size,
                num_workers=self.args.n_dataloader_workers, drop_last=True, device=self.device, train=1, seed=self.args.seed,
                shuffle=True, replacement=self.args.sample_with_replacement,
                distributed=self.world_size > 1
            )
        else:
            self.bn_dataloader = None
    
    
    @classmethod
    def init_get_metrics(cls, self):
        def metric_acc(output, target):
            return count_correct(output, target, topk=(1, 5))
        
        return [metric_acc], ["acc1", "acc5"]
        

    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser):
        super().add_argparse_args(parser)

        parser.set_defaults(**{
            'lr_sche_type': 'multistep',
            'base_batch_size_for_lr': 256,
            'autocast_dtype': 'float16',
            'grad_scaler': 1,
            'arch_lib': ['klib.imagenet'],
        })

        group = parser.add_argument_group('ImageNet special')
        group.add_argument('--dataloader-backend', type=str, default='ffcv')
        group.add_argument('--n-dataloader-workers', type=int, default=6)
        group.add_argument('--num-classes', type=int, default=1000)
        group.add_argument('--criterion', type=str, default='ce')
        group.add_argument('--sample-with-replacement', type=int, default=0)


    @classmethod
    def input_size(cls, self):
        return (3, 224, 224)
