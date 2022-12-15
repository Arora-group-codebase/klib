
import torch
from torch import nn
from typing import Set
from klib.kmodel import KModel


__all__ = ['KModelSIA']


class KModelSIA(KModel):
    """Scale-invariance-aware KModel
    """

    si_group: Set[nn.Parameter]
    
    
    def __init__(self, model, si_group, criterion, metrics, metric_names):
        super().__init__(model, criterion, metrics, metric_names)
        self.si_group = set(si_group)
    
    
    @torch.no_grad()
    def log_param_stats(self, log):
        si_norm2 = 0.
        nsi_norm2 = 0.
        for name, param in self.model.named_parameters():
            norm2 = (param.data ** 2).sum().float().item()

            if param in self.param_name_dict:
                name = self.param_name_dict[param]
            
            if param in self.si_group:
                log[f'norm/si/{name}'] = norm2 ** 0.5
                si_norm2 += norm2
            else:
                log[f'norm/nsi/{name}'] = norm2 ** 0.5
                nsi_norm2 += norm2
        
        log[f'norm/si/all'] = si_norm2 ** 0.5
        log[f'norm/nsi/all'] = nsi_norm2 ** 0.5
        log[f'norm/all'] = (si_norm2 + nsi_norm2) ** 0.5


    @torch.no_grad()
    def log_grad_stats(self, log):
        si_norm2 = 0.
        nsi_norm2 = 0.
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm2 = (param.grad ** 2).sum().float().item()

                if param in self.param_name_dict:
                    name = self.param_name_dict[param]
                
                if param in self.si_group:
                    log[f'gnorm/si/{name}'] = norm2 ** 0.5
                    si_norm2 += norm2
                else:
                    log[f'gnorm/nsi/{name}'] = norm2 ** 0.5
                    nsi_norm2 += norm2
        
        log[f'gnorm/si/all'] = si_norm2 ** 0.5
        log[f'gnorm/nsi/all'] = nsi_norm2 ** 0.5
        log[f'gnorm/all'] = (si_norm2 + nsi_norm2) ** 0.5
    