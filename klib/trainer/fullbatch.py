from functools import cached_property
import torch
from .base import BaseTrainer
import wandb
from klib.kdataloader import KDataLoader

__all__ = ['FullBatchTrainer']

class FullBatchTrainer(BaseTrainer):
    
    train_dataloader: KDataLoader
    test_dataloader: KDataLoader
    bn_dataloader: KDataLoader
    optimizer: torch.optim.Optimizer

    def post_init_check(self):
        assert self.train_dataloader.dataset_size() % self.total_batch_size == 0
        

    def run_epoch(self, train=True):
        self.on_train_step_start()
        
        self.index_in_epoch = 0

        cur_stats = self.prop_step(self.train_dataloader.enum())

        self.log_train_step(cur_stats)

        if train and cur_stats is not None:
            self.optimizer_step()

        self.on_train_step_end()
    
    
    @cached_property
    def steps_per_epoch(self) -> int:
        return 1


    @cached_property
    def n_grad_accumu(self) -> int:
        return len(self.train_dataloader)
