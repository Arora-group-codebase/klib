import torch
import torch.distributed as dist

from klib.kmodel import KModel
from klib.kdataloader import KDataLoader
from klib.kgrad_scaler import GradScaleTooLargeError
from klib.train_utils import get_flat_tensor_from_tensor_sequence, set_flat_tensor_to_tensor_sequence
from .base import BaseTrainer, BaseTrainerSetup
import wandb
from functools import cached_property
import math

import time

__all__ = ['StochasticTrainer']


class StochasticTrainer(BaseTrainer):
    
    train_dataloader: KDataLoader
    test_dataloader: KDataLoader
    bn_dataloader: KDataLoader
    optimizer: torch.optim.Optimizer

    train_stats = 0

    def run_epoch(self, train=True):
        n_steps = None if train else 1
        for self.index_in_epoch, (inputs, targets) in self.train_dataloader.enum(n_steps):
            if not self.stop_now:
                self.train_step(inputs, targets, train)

    
    def train_step(self, inputs, targets, train):
        # go from step_ctr to step_ctr + 1

        self.on_train_step_start()

        self.debug_log(f'train step #{self.step_ctr} start')
        
        def step():
            train_step_kwargs = dict(
                autocast=self.autocast,
                grad_upscale=self.kgrad_scaler.scale if self.kgrad_scaler is not None else self.args.grad_upscale
            )
            if self.n_grad_accumu == 1:
                self.optimizer.zero_grad()
                cur_stats = self.kmodel.train_step(inputs, targets, **train_step_kwargs)
            else:
                cur_stats = 0
                subinputs = inputs.tensor_split(self.n_grad_accumu)
                subtargets = targets.tensor_split(self.n_grad_accumu)

                assert len(subinputs) == self.n_grad_accumu
                assert len(subtargets) == self.n_grad_accumu
                assert all(subinputs[i].shape[0] == self.physical_batch_size for i in range(self.n_grad_accumu))
                assert all(subtargets[i].shape[0] == self.physical_batch_size for i in range(self.n_grad_accumu))
                
                train_step_kwargs['grad_upscale'] /= self.n_grad_accumu

                if self.args.autocast_dtype == 'float16':
                    assert self.args.ddp_backend != 'torch'
                    flat = 0
                    for i in range(self.n_grad_accumu):
                        self.debug_log(f'train step #{self.step_ctr} grad accumu #{i}')
                        self.optimizer.zero_grad()
                        cur_stats += self.kmodel.train_step(subinputs[i], subtargets[i], **train_step_kwargs)
                        flat += get_flat_tensor_from_tensor_sequence(self.kmodel.grads())
                    set_flat_tensor_to_tensor_sequence(flat, self.kmodel.grads())
                else:
                    with self.kmodel.no_ddp_sync():
                        for i in range(self.n_grad_accumu - 1):
                            self.debug_log(f'train step #{self.step_ctr} grad accumu #{i}')
                            cur_stats += self.kmodel.train_step(subinputs[i], subtargets[i], **train_step_kwargs)
                    self.debug_log(f'train step #{self.step_ctr} grad accumu #last')
                    cur_stats += self.kmodel.train_step(subinputs[-1], subtargets[-1], **train_step_kwargs)

            self.post_process_grad()

            if self.world_size > 1:
                self.debug_log(f'train step #{self.step_ctr} sync cur stats')
                dist.reduce(cur_stats, 0)
            
            return cur_stats
        
        for t in range(self.args.grad_scaler_max_retries):
            try:
                cur_stats = step()
                break
            except GradScaleTooLargeError:
                pass
            if t == self.args.grad_scaler_max_retries - 1:
                if self.rank == 0:
                    print("ERROR: cannot find a good grad scaling")
                cur_stats = None
                self.stop_now = True
        
        if self.rank == 0:
            self.debug_log(f'train step #{self.step_ctr} log step')
            if self.kgrad_scaler is not None:
                self.step_log.update({ 'train/grad_upscale': self.kgrad_scaler.scale })
            if cur_stats is not None:
                self.kmodel.log_grad_stats(self.step_log) # if ddp_backend == 'avg-model', then it only logs the grads for rank 0
                self.kmodel.log_avg_step_stats(self.step_log, 'train', cur_stats)
                self.train_stats += cur_stats
            wandb.log(self.step_log, step=self.step_ctr)

        if train and cur_stats is not None:
            self.optimizer_step()

        self.on_train_step_end()

        self.debug_log(f'train step #{self.step_ctr} end')
    
    
    def on_train_step_start(self):
        self.kscheduler.on_train_step_start(index_in_epoch=self.index_in_epoch)
        
        if self.rank == 0:
            self.step_log = {
                "train/step": self.step_ctr,
                "time/total": time.time() - self.start_time,
                "time/dataloader": KDataLoader.total_load_time,
                "time/train": KModel.total_train_time,
                "time/eval": KModel.total_eval_time,
                "time/update_bn": KModel.total_update_bn_time,
            }
            self.kmodel.log_param_stats(self.step_log)
            if math.isnan(self.step_log['norm/all']) or math.isinf(self.step_log['norm/all']):
                print("ERROR: NaN detected!")
                self.stop_now = True
            self.kscheduler.log('train', self.step_log)
        
        if self.index_in_epoch == 0:
            if self.rank == 0:
                self.step_log["epoch"] = self.epoch
                if isinstance(self.train_stats, torch.Tensor):
                    self.kmodel.log_avg_step_stats(self.step_log, 'train_avg', self.train_stats)
                self.train_stats = 0

            if self.args.eval_freq is not None and self.epoch % self.args.eval_freq == 0:
                self.on_evaluate_start()
                self.evaluate()
    

    def on_evaluate_start(self):
        if self.bn_dataloader is not None:
            self.estimate_bn_stats()
    
    
    @torch.no_grad()
    def estimate_bn_stats(self):
        for idx, (images, targets) in self.bn_dataloader.enum(self.args.bn_batches // self.world_size):
            self.debug_log(f'estimating BN: {idx}')
            self.kmodel.update_bn(idx, images, autocast=self.autocast)

        if self.world_size > 1:
            flat = get_flat_tensor_from_tensor_sequence(self.kmodel.bn_buffers())
            dist.all_reduce(flat)
            flat /= self.world_size
            set_flat_tensor_to_tensor_sequence(flat, self.kmodel.bn_buffers())


    @torch.no_grad()
    def evaluate(self):
        total_step_stats = 0
        for idx, (inputs, targets) in self.test_dataloader.enum():
            self.debug_log(f'eval step: {idx}')
            total_step_stats += self.kmodel.eval_step(inputs, targets, autocast=self.autocast)
        
        if self.world_size > 1:
            dist.reduce(total_step_stats, 0)
        
        if self.rank == 0:
            self.kmodel.log_avg_step_stats(self.step_log, 'test', total_step_stats)
    
    
    def on_train_step_end(self):
        self.kscheduler.on_train_step_end(self.index_in_epoch)
    
    
    @cached_property
    def steps_per_epoch(self) -> int:
        return len(self.train_dataloader)
