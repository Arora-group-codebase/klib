from math import prod
import torch
from torch import nn
from torch.nn import functional as F
from klib.homo.kmodel import KModelHomo
from klib.train_utils import get_activation
    

class HomoMLP(nn.Module):
    def __init__(self, L=None, dimD=None, dimO=None, dimH=None, init=None, activation=None, first_layer_bias=False, **kwargs):
        super().__init__()

        self.L = L
        self.dimD = dimD
        self.dimH = dimH
        self.dimO = dimO

        self.layers = []
        seq = []
        dimLast = self.dimD
        for k in range(self.L - 1):
            l = nn.Linear(dimLast, self.dimH, bias=(k == 0 and first_layer_bias))
            self.layers.append(l)
            seq.append(l)
            dimLast = self.dimH
            seq.append(get_activation(activation))
        self.features = nn.Sequential(*seq)

        self.layers.append(nn.Linear(dimLast, self.dimO, bias=False))
        self.final_layer = self.layers[-1]

        if init == 'default':
            pass
        elif init == 'he':
            for l in self.layers:
                torch.nn.init.kaiming_normal_(l.weight.data, nonlinearity=activation)
        elif init.startswith('sm'):
            for name, param in self.named_parameters():
                param.data.normal_(0, 0.1 ** int(init[2:]))
        else:
            raise NotImplementedError()
    
    def forward(self, x):
        x = self.features(x)
        out = self.final_layer(x)
        if self.dimO == 1:
            out = out.squeeze(dim=1)
        return out


def homo_mlp(args):
    net = HomoMLP(L=args.depth, dimD=args.dimD, dimH=args.dimH, dimO=args.dimO, init=args.init, activation=args.activation, first_layer_bias=args.first_layer_bias)
    homo_group = []
    for i, l in enumerate(net.layers):
        g = {
            'name': f"L{i+1}",
            'params': [l.weight],
            'degree': 1,
        }
        if l.bias is not None:
            g['params'].append(l.bias)
        homo_group.append(g)
    return KModelHomo, net, homo_group


"""
class KHomoMLPMultiCE(KHomoMLP):
    def __init__(self, log_loss=False, **kwargs):
        if not log_loss:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = LogCrossEntropyLoss()

        super().__init__(criterion, [AccuracySum(), MultiLabelMargin()], ['acc', 'margin'], **kwargs)
    
    @torch.no_grad()
    def log_param_stats(self, log):
        super().log_param_stats(log)
        
        log['norm/wb1'] = log['norm/w1'] if 'norm/b1' not in log else (log['norm/w1'] ** 2 + log['norm/b1'] ** 2) ** 0.5
        log['norm/prod'] = prod((log[f'norm/w{k+1}'] if k > 0 else log['norm/wb1'] for k in range(self.L)))


    @torch.no_grad()
    def log_avg_step_stats(self, log, type, stats):
        super().log_avg_step_stats(log, type, stats)
        log.update({
            f'{type}/margin_over_norm_all': log[f'{type}/margin'] / log['norm/all'] ** self.L,
            f'{type}/margin_over_norm_prod': log[f'{type}/margin'] / log['norm/prod'],
        })
"""
