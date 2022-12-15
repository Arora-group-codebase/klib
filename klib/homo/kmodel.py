import torch
from klib.kmodel import KModel


class KModelHomo(KModel):
    def __init__(self, model, homo_group, criterion, metrics, metric_names):
        super().__init__(model, criterion, metrics, metric_names)
        self.homo_group = homo_group
        
        rest = set(model.parameters())
        for g in self.homo_group:
            for p in g['params']:
                rest.remove(p)
        assert not rest
    

    @torch.no_grad()
    def log_param_stats(self, log):
        all_norm2 = 0.
        norm_prod = 1.
        for g in self.homo_group:
            for g in self.homo_group:
                norm2 = 0.
                for p in g['params']:
                    norm2 += (p.data ** 2).sum().float().item()
                log[f"norm/h{g['degree']}/{g['name']}"] = norm2 ** 0.5
                all_norm2 += norm2
                norm_prod *= norm2 ** g['degree']
        log[f'norm/all'] = all_norm2 ** 0.5
        log[f'norm/prod'] = norm_prod ** 0.5


    @torch.no_grad()
    def log_grad_stats(self, log):
        all_norm2 = 0.
        for g in self.homo_group:
            for g in self.homo_group:
                norm2 = 0.
                for p in g['params']:
                    if p.grad is not None:
                        norm2 += (p.grad ** 2).sum().float().item()
                log[f"gnorm/h{g['degree']}/{g['name']}"] = norm2 ** 0.5
                all_norm2 += norm2
        log[f'gnorm/all'] = all_norm2 ** 0.5
