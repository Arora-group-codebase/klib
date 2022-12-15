import torch
from torch import nn
import contextlib
from typing import Iterable

import klib.train_utils

import time

class KModel:
    
    model: nn.Module
    criterion: nn.Module
    metrics: Iterable[callable]
    metric_names: Iterable[str]
    param_name_dict: dict

    total_train_time = 0
    total_eval_time = 0
    total_update_bn_time = 0

    def __init__(self, model, criterion, metrics, metric_names):
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.metric_names = metric_names
        self.param_name_dict = {}
    
    
    def train_step(self, inputs, targets, *, autocast, grad_upscale=1):
        last = time.time()
        res = self._train_step(inputs, targets, autocast=autocast, grad_upscale=grad_upscale)
        KModel.total_train_time += time.time() - last
        return res


    def _train_step(self, inputs, targets, *, autocast, grad_upscale=1):
        if not self.model.training: # optimize for speed
            self.model.train()

        with autocast():
            output = self.model(inputs)

        cur_loss = self.criterion(output, targets)
        cur_rescaled_loss = cur_loss if grad_upscale == 1 else cur_loss * grad_upscale
        cur_rescaled_loss.backward()

        with torch.no_grad():
            cur_metrics = []
            for m in self.metrics:
                mval = m(output, targets)
                if isinstance(mval, (list, tuple)):
                    cur_metrics.extend(mval)
                else:
                    cur_metrics.append(mval)

            return torch.stack([
                *cur_metrics, cur_loss * inputs.shape[0],
                torch.as_tensor(inputs.shape[0], dtype=cur_loss.dtype, device=cur_loss.device)
            ])
        

    @torch.no_grad()
    def eval_step(self, inputs, targets, *, autocast):
        last = time.time()
        res = self._eval_step(inputs, targets, autocast=autocast)
        KModel.total_eval_time += time.time() - last
        return res


    def _eval_step(self, inputs, targets, *, autocast):
        if self.model.training: # optimize for speed
            self.model.eval()

        with autocast():
            output = self.model(inputs)
            cur_loss = self.criterion(output, targets)

            cur_metrics = []
            for m in self.metrics:
                mval = m(output, targets)
                if isinstance(mval, (list, tuple)):
                    cur_metrics.extend(mval)
                else:
                    cur_metrics.append(mval)

        return torch.stack([
            *cur_metrics, cur_loss * inputs.shape[0],
            torch.as_tensor(inputs.shape[0], dtype=cur_loss.dtype, device=cur_loss.device)
        ])


    def has_bn(self):
        for m in self.model.modules():
            if klib.train_utils.is_bn(m):
                return True
        return False


    @torch.no_grad()
    def update_bn(self, idx, images, *, autocast):
        last = time.time()

        if not self.model.training: # optimize for speed
            self.model.train()

        for m in self.model.modules():
            if klib.train_utils.is_bn(m):
                m.momentum = 1 / (1 + idx)

        with autocast():
            self.model(images)
        
        KModel.total_update_bn_time += time.time() - last
    

    @torch.no_grad()
    def log_param_stats(self, log):
        all_norm2 = 0.
        for name, param in self.model.named_parameters():
            norm2 = (param.data ** 2).sum().float().item()

            if param in self.param_name_dict:
                name = self.param_name_dict[param]
            
            log[f'norm/{name}'] = norm2 ** 0.5
            all_norm2 += norm2
        
        log[f'norm/all'] = all_norm2 ** 0.5


    @torch.no_grad()
    def log_grad_stats(self, log):
        all_norm2 = 0.
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm2 = (param.grad ** 2).sum().float().item()

                if param in self.param_name_dict:
                    name = self.param_name_dict[param]

                log[f'gnorm/{name}'] = norm2 ** 0.5
                all_norm2 += norm2
        
        log[f'gnorm/all'] = all_norm2 ** 0.5


    @torch.no_grad()
    def log_avg_step_stats(self, log, type, stats):
        avg_stats = stats[:-1] / stats[-1]
        assert avg_stats.shape[0] == len(self.metric_names) + 1
        log.update({
            **{f"{type}/{self.metric_names[k]}": avg_stats[k] for k in range(len(self.metric_names))},
            f"{type}/loss": avg_stats[-1],
        })
    
    
    def grads(self):
        for p in self.model.parameters():
            if p.requires_grad:
                yield p.grad
    
    
    def trainable_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                yield p


    def bn_buffers(self):
        for name, buffer in self.model.named_buffers():
            if name.endswith("mean") or name.endswith("var"):
                yield buffer


    @torch.no_grad()
    def constrain_param_norm(self, constraint):
        if constraint < 0:
            return
        
        all_norm2 = 0.
        for name, param in self.model.named_parameters():
            norm2 = (param.data ** 2).sum()
            all_norm2 += norm2
        
        all_norm = all_norm2 ** 0.5
        if all_norm > constraint:
            factor = constraint / all_norm
            for name, param in self.model.named_parameters():
                param *= factor


    def no_ddp_sync(self):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.no_sync()
        else:
            return contextlib.nullcontext()

