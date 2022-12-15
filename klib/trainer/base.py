import torch
import torch.distributed as dist
import os, sys
from klib.kdataloader import KDataLoader
from klib.kmodel import KModel
from klib.ksche import KScheduler, KConstScheduler
from klib.kgrad_scaler import GradScaleTooLargeError, KGradScaler
import numpy as np
import random
import yaml
import argparse
from pathlib import Path
import torchsummary
from torch.optim import Optimizer
import torch.cuda
import torch.cuda.amp
import wandb
import signal
import math

from klib.train_utils import load_builtin_optimizer, load_builtin_model, load_builtin_criterion, TORCH_DTYPES
from klib.train_utils import get_flat_tensor_from_tensor_sequence, set_flat_tensor_to_tensor_sequence

import time

__all__ = ['BaseTrainerSetup', 'BaseTrainer']


class BaseTrainerSetup:

    @classmethod
    def init(cls, self):
        cls.pre_init(self)
        cls.init_data(self)
        cls.init_kmodel(self)
        cls.init_extra_data(self)
        cls.init_optimizer(self)
        cls.init_scheduler(self)
        cls.post_init(self)
    
    
    @classmethod
    def pre_init(cls, self):
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed + 10)
        random.seed(self.args.seed + 20)

        self.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        if self.world_size > 1:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ['LOCAL_RANK']) # Is equal to rank in single machine setting

            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")

            dist.init_process_group(backend='nccl', init_method=self.args.dist_url,
                                    world_size=self.world_size, rank=self.rank)
            dist.barrier()
        else:
            self.rank = self.local_rank = 0
            self.device = torch.device('cuda')
        
        
        if 'total_batch_size' in self.args.__dict__:
            self.total_batch_size = self.args.total_batch_size
            assert self.total_batch_size % self.world_size == 0
            self.local_batch_size = self.total_batch_size // self.world_size
            if self.args.physical_batch_size is None:
                self.args.physical_batch_size = self.local_batch_size
            self.physical_batch_size = self.args.physical_batch_size
            assert self.local_batch_size % self.physical_batch_size == 0
            if self.args.bn_batch_size is None:
                self.args.bn_batch_size = self.physical_batch_size
            self.bn_batch_size = self.args.bn_batch_size
            assert self.physical_batch_size % self.bn_batch_size == 0

        if self.args.grad_scaler:
            self.args.grad_upscale *= self.n_grad_accumu
        else:
            self.args.grad_upscale = self.n_grad_accumu


    @classmethod
    def init_data(cls, self):
        """Initialize the data
        """
        raise NotImplementedError()
    
    
    @classmethod
    def init_extra_data(cls, self):
        pass
    

    @classmethod
    def init_kmodel(cls, self):
        """Initialize the kmodel
        """
        mcls, model, *extra = cls.init_get_model(self)
        
        model = model.to(self.device)
        self.debug_log(f"model device: {self.device}")
        model.eval()
        if self.rank == 0:
            torchsummary.summary(model, cls.input_size(self))
            
        model = cls.init_model_post_process(self, model)
        self.kmodel = mcls(model, *extra, cls.init_get_criterion(self), *cls.init_get_metrics(self))
    
    
    @classmethod
    def init_get_model(cls, self):
        return load_builtin_model(self.args)
    
    
    @classmethod
    def init_model_post_process(cls, self, model):
        if self.world_size > 1 and self.args.ddp_backend == 'torch':
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], broadcast_buffers=False)
        model.eval()
        return model
    
    
    @classmethod
    def init_get_criterion(cls, self):
        return load_builtin_criterion(self.args)
    
    
    @classmethod
    def init_get_metrics(cls, self):
        """Return metrics and metric names for initializing the trainer

        Returns:
            List[callable], List[str]: metrics and metric names
        """
        return [], []
    

    @classmethod
    def init_optimizer(cls, self):
        """Initialize the optimizer
        """
        self.optimizer = load_builtin_optimizer(self.kmodel.model, self.args)
        if self.args.autocast_dtype != 'float16':
            self.args.grad_scaler = 0
            
        if self.args.grad_scaler:
            self.kgrad_scaler = KGradScaler(
                init_scale=self.args.grad_upscale,
                growth_factor=self.args.grad_scaler_growth_factor,
                backoff_factor=self.args.grad_scaler_backoff_factor,
                growth_interval=self.args.grad_scaler_growth_interval
            )
    

    @classmethod
    def init_scheduler(cls, self):
        """Initialize the scheduler (for LR, WD, etc.)
        """
        self.kscheduler = KConstScheduler(self.optimizer)
    

    @classmethod
    def post_init(cls, self):
        pass


    @classmethod
    def input_size(cls, self):
        raise NotImplementedError()
    
    
    @classmethod
    def default_args_dict(cls) -> dict:
        return {
            'ddp_backend': 'avg-grad',
            'autocast_dtype': 'float32',
            'grad_scaler': 0,
            'grad_upscale': 65536,
            'grad_scaler_max_retries': 50,
            'grad_scaler_growth_factor': 2,
            'grad_scaler_backoff_factor': 0.5,
            'grad_scaler_growth_interval': 100,
            'train_step_log_freq': 1,
        }


class BaseTrainer:

    world_size: int

    kmodel: KModel = None
    kscheduler: KScheduler = None
    kgrad_scaler: KGradScaler = None
    
    optimizer: Optimizer

    device: torch.device

    index_in_epoch: int = 0
    step_log: dict

    start_time: float

    stop_now: bool = False

    n_grad_accumu: int
    total_batch_size: int
    local_batch_size: int
    physical_batch_size: int
    bn_batch_size: int
    train_stats = 0


    def __init__(self, args, setup: BaseTrainerSetup):
        opt = setup.default_args_dict()
        opt.update(yaml.load(open(args.recipe_pth), Loader=yaml.FullLoader))
        for k, v in vars(args).items():
            if v is not None or k not in opt:
                opt[k] = v
        self.args = argparse.Namespace(**opt)
        self.args.save_path = './check_point/' + Path(args.recipe_pth).stem + f'-{args.seed}'
        setup.init(self)
        self.post_init_check()

        self.hooks = {
            'extra_train_step_log': [],
        }
    
    
    def post_init_check(self):
        pass
    

    def run(self):
        self.start_time = time.time()
        
        self.resume_step()
        for nepoch in range(self.args.epochs + 1):
            self.on_epoch_start()
            self.run_epoch(train=(nepoch < self.args.epochs))
            if self.stop_now:
                break
        
        if self.world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        
        signal.signal(signal.SIGALRM, lambda: sys.exit(1))
        signal.alarm(60)
        if self.rank == 0:
            wandb.finish()
            signal.alarm(0)
        

    def resume_step(self):
        if self.args.resume_pth is not None:
            self.kmodel.model.load_state_dict(torch.load(self.args.resume_pth, map_location=self.device))
            epoch = self.args.resume
        else:
            epoch = 0
        step_ctr = self.epoch * self.steps_per_epoch

        self.kscheduler.resume(epoch, step_ctr)
    

    def on_epoch_start(self):
        self.save_step()


    @torch.no_grad()
    def save_step(self):
        if self.rank == 0:
            if self.args.save_latest:
                torch.save(self.kmodel.model.state_dict(), os.path.join(self.args.save_pth, "latest.pt"))
            if self.args.save_freq != -1 and self.epoch % self.args.save_freq == 0:
                torch.save(self.kmodel.model.state_dict(), os.path.join(self.args.save_pth, f"epoch{self.epoch}.pt"))
                self.debug_log(f'epoch #{self.epoch} saved')
    
    
    def run_epoch(self, train=True):
        raise NotImplementedError()
    
    
    def autocast(self):
        return torch.cuda.amp.autocast(dtype=TORCH_DTYPES[self.args.autocast_dtype])
    
    
    @torch.no_grad()
    def post_process_grad(self):
        if self.world_size > 1 and self.args.ddp_backend == 'avg-grad':
            flat = get_flat_tensor_from_tensor_sequence(self.kmodel.grads())
            dist.all_reduce(flat)
            flat /= self.world_size
            set_flat_tensor_to_tensor_sequence(flat, self.kmodel.grads())
        
        if self.kgrad_scaler is not None:
            self.kgrad_scaler.unscale_(self.optimizer)
        else:
            for g in self.optimizer.param_groups:
                for p in g['params']:
                    if p.grad is not None:
                        p.grad /= self.args.grad_upscale
    
    
    def prop_step(self, batch):
        if isinstance(batch, tuple):
            if self.n_grad_accumu == 1:
                batch = [(0,) + batch]
            else:
                inputs, targets = batch
                subinputs = inputs.tensor_split(self.n_grad_accumu)
                subtargets = targets.tensor_split(self.n_grad_accumu)
                assert len(subinputs) == self.n_grad_accumu
                assert len(subtargets) == self.n_grad_accumu
                assert all(subinputs[i].shape[0] == self.physical_batch_size for i in range(self.n_grad_accumu))
                assert all(subtargets[i].shape[0] == self.physical_batch_size for i in range(self.n_grad_accumu))
                batch = [(k, subinputs[k], subtargets[k]) for k in range(self.n_grad_accumu)]
        else:
            assert self.args.autocast_dtype != 'float16' or self.n_grad_accumu == 1


        def step():
            train_step_kwargs = dict(
                autocast=self.autocast,
                grad_upscale=self.kgrad_scaler.scale if self.kgrad_scaler is not None else self.args.grad_upscale
            )

            if self.n_grad_accumu == 1:
                self.optimizer.zero_grad()
                for i, inputs, targets in batch:
                    assert i == 0
                    cur_stats = self.kmodel.train_step(inputs, targets, **train_step_kwargs)
            else:
                cur_stats = 0
                train_step_kwargs['grad_upscale'] /= self.n_grad_accumu

                if self.args.autocast_dtype == 'float16':
                    assert self.args.ddp_backend != 'torch'
                    flat = 0
                    for i, inputs, targets in batch:
                        self.debug_log(f'train step #{self.step_ctr} grad accumu #{i}')
                        self.optimizer.zero_grad()
                        cur_stats += self.kmodel.train_step(inputs, targets, **train_step_kwargs)
                        flat += get_flat_tensor_from_tensor_sequence(self.kmodel.grads())
                    set_flat_tensor_to_tensor_sequence(flat, self.kmodel.grads())
                else:
                    with self.kmodel.no_ddp_sync():
                        for i, inputs, targets in batch:
                            self.debug_log(f'train step #{self.step_ctr} grad accumu #{i}')
                            if i < self.n_grad_accumu - 1:
                                cur_stats += self.kmodel.train_step(inputs, targets, **train_step_kwargs)
                    self.debug_log(f'train step #{self.step_ctr} grad accumu #last')
                    cur_stats += self.kmodel.train_step(inputs, targets, **train_step_kwargs)

            self.post_process_grad()

            if self.world_size > 1:
                self.debug_log(f'train step #{self.step_ctr} sync cur stats')
                dist.reduce(cur_stats, 0)
            
            return cur_stats

        
        for t in range(self.args.grad_scaler_max_retries):
            try:
                return step()
            except GradScaleTooLargeError:
                pass
        
        if self.rank == 0:
            print("ERROR: cannot find a good grad scaling")
        self.stop_now = True
        return None

    
    @torch.no_grad()
    def optimizer_step(self):
        self.debug_log(f'train step #{self.step_ctr} optimizer step')
        self.optimizer.step()

        if self.world_size > 1 and self.args.ddp_backend == 'avg-model':
            flat = get_flat_tensor_from_tensor_sequence(self.kmodel.trainable_params())
            torch.cuda.synchronize()
            dist.all_reduce(flat)
            flat /= self.world_size
            set_flat_tensor_to_tensor_sequence(flat, self.kmodel.trainable_params())

        if self.kgrad_scaler is not None:
            self.kgrad_scaler.update()


    def need_to_log(self):
        if self.step_ctr % self.args.train_step_log_freq == 0:
            return True
        if self.index_in_epoch == 0 and self.args.eval_freq is not None and self.epoch % self.args.eval_freq == 0:
            return True
        return False


    def on_train_step_start(self):
        self.debug_log(f'train step #{self.step_ctr} start')

        self.kscheduler.on_train_step_start(index_in_epoch=self.index_in_epoch)
        
        if self.need_to_log() and self.rank == 0:
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
                if self.need_to_log():
                    self.step_log["epoch"] = self.epoch
                    if isinstance(self.train_stats, torch.Tensor):
                        self.kmodel.log_avg_step_stats(self.step_log, 'train_avg', self.train_stats)

                self.train_stats = 0

            if self.args.eval_freq is not None and self.epoch % self.args.eval_freq == 0:
                self.on_evaluate_start()
                self.evaluate()
                self.on_evaluate_end()
    
    
    def log_train_step(self, cur_stats):
        if self.rank == 0:
            if cur_stats is not None:
                self.train_stats += cur_stats

            if self.need_to_log():
                self.debug_log(f'train step #{self.step_ctr} log step')
                if self.kgrad_scaler is not None:
                    self.step_log.update({ 'train/grad_upscale': self.kgrad_scaler.scale })
                if cur_stats is not None:
                    self.kmodel.log_grad_stats(self.step_log) # if ddp_backend == 'avg-model', then it only logs the grads for rank 0
                    self.kmodel.log_avg_step_stats(self.step_log, 'train', cur_stats)
                self._call_hook('extra_train_step_log')
                wandb.log(self.step_log, step=self.step_ctr)
    
    
    def on_train_step_end(self):
        self.kscheduler.on_train_step_end(self.index_in_epoch)
        self.debug_log(f'train step #{self.step_ctr} end')
        
    
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
    
    
    def on_evaluate_end(self):
        pass
    
    
    def register_hook(self, name, func):
        self.hooks[name].append(func)
    
    
    def _call_hook(self, name):
        for func in self.hooks[name]:
            func(self)
    
    
    def debug_log(self, s):
        if self.rank == 0 and self.args.debug:
            print(s)
    

    @property
    def epoch(self) -> int:
        return self.kscheduler.epoch


    @property
    def step_ctr(self) -> int:
        return self.kscheduler.step_ctr
    

    @property
    def steps_per_epoch(self) -> int:
        raise NotImplementedError()
    
    
    @property
    def n_grad_accumu(self) -> int:
        return self.local_batch_size // self.physical_batch_size
