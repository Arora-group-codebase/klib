import torch

from klib.kdataloader import KDataLoader
from klib.train_utils import get_flat_tensor_from_tensor_sequence, set_flat_tensor_to_tensor_sequence
from .base import BaseTrainer
import wandb
from functools import cached_property

__all__ = ['StochasticTrainer']


class StochasticTrainer(BaseTrainer):
    
    train_dataloader: KDataLoader
    test_dataloader: KDataLoader
    bn_dataloader: KDataLoader
    optimizer: torch.optim.Optimizer


    def run_epoch(self, train=True):
        n_steps = None if train else 1
        for self.index_in_epoch, (inputs, targets) in self.train_dataloader.enum(n_steps):
            if not self.stop_now:
                self.train_step(inputs, targets, train)

    
    def train_step(self, inputs, targets, train):
        # go from step_ctr to step_ctr + 1

        self.on_train_step_start()

        cur_stats = self.prop_step((inputs, targets))

        self.log_train_step(cur_stats)

        if train and cur_stats is not None:
            self.optimizer_step()

        self.on_train_step_end()

    
    @cached_property
    def steps_per_epoch(self) -> int:
        return len(self.train_dataloader)
