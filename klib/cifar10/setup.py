import klib.trainer
from klib.ksche import KMultiStepScheduler
from .data import get_kdataloader
from klib.train_utils import count_correct

from . import arch

__all__ = ['CIFAR10TrainerSetup']


VAL_B = 1000


class CIFAR10TrainerSetup(klib.trainer.BaseTrainerSetup):

    @classmethod
    def init_data(cls, self):
        self.train_dataloader = get_kdataloader(
            backend=self.args.dataloader_backend,
            data_pth=self.args.data_pth, batch_size=self.local_batch_size,
            num_workers=self.args.n_dataloader_workers, drop_last=True, device=self.device, train=1, seed=self.args.seed,
            crop=self.args.aug_crop, hflip=self.args.aug_hflip,
            distributed=self.world_size > 1
        )

        self.test_dataloader = get_kdataloader(
            backend=self.args.dataloader_backend,
            data_pth=self.args.data_pth, batch_size=VAL_B,
            num_workers=self.args.n_dataloader_workers, drop_last=False, device=self.device, train=0, seed=self.args.seed,
            crop=False, hflip=False,
            distributed=self.world_size > 1
        )
    
    
    @classmethod
    def init_get_model(cls, self):
        return arch.__dict__[self.args.arch](self.args)
    
    
    @classmethod
    def init_extra_data(cls, self):
        if self.kmodel.has_bn():
            self.bn_dataloader = get_kdataloader(
                backend=self.args.dataloader_backend,
                data_pth=self.args.data_pth, batch_size=self.bn_batch_size,
                num_workers=self.args.n_dataloader_workers, drop_last=True, device=self.device, train=1, seed=self.args.seed,
                crop=self.args.aug_crop, hflip=self.args.aug_hflip,
                distributed=self.world_size > 1
            )
    

    @classmethod
    def init_scheduler(cls, self):
        scale = self.total_batch_size / 128
        rescaled_lr = self.args.lr * scale

        if self.args.warmup:
            warmup_steps = self.args.warmup_epochs * self.steps_per_epoch
            if self.args.warmup_start_lr is None:
                self.args.warmup_start_lr = self.args.lr
            warmup_sche = [self.args.warmup_start_lr * (1 + (scale - 1) * (t / (warmup_steps - 1))) for t in range(warmup_steps)]
        else:
            warmup_sche = [rescaled_lr]
        
        self.kscheduler = KMultiStepScheduler(
            self.optimizer,
            warmup_sche=warmup_sche,
            lrdecay_sche=self.args.lrdecay_sche,
            gamma=self.args.gamma
        )

    
    @classmethod
    def init_get_metrics(cls, self):
        def metric_acc(output, target):
            return count_correct(output, target, topk=(1,))
        
        return [metric_acc], ["acc"]
        

    @classmethod
    def default_args_dict(cls) -> dict:
        return {
            **super().default_args_dict(),
            "num_classes": 10,
            "criterion": "ce"
        }

    @classmethod
    def input_size(cls):
        return (3, 32, 32)
