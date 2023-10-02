import torch

from klib.kdataloader import KDataLoader
from .base import BaseTrainer
from functools import cached_property

__all__ = ['StochasticTrainer']


class StochasticTrainer(BaseTrainer):
    
    train_dataloader: KDataLoader
    test_dataloader: KDataLoader
    bn_dataloader: KDataLoader
    optimizer: torch.optim.Optimizer

    
    def post_init_check(self):
        if self.args.steps_per_epoch == -1:
            self.args.steps_per_epoch = len(self.train_dataloader)


    def run_epoch(self, train=True):
        n_steps = self.steps_per_epoch if train else 1
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
