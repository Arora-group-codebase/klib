from math import prod
import torch
from torch import nn
from torch.nn import functional as F
from klib.kmodel import KModel
from klib.train_utils import get_activation, get_norm1d
    

class MLPN(nn.Module):
    def __init__(self, L=None, dimD=None, dimO=None, dimH=None, init=None, activation=None, first_layer_bias=False, norm_layer=None, **kwargs):
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
            seq.append(norm_layer(dimLast))
            seq.append(get_activation(activation))
        self.features = nn.Sequential(*seq)

        self.layers.append(nn.Linear(dimLast, self.dimO, bias=False))
        self.final_layer = self.layers[-1]

        if init == 'default':
            pass
        elif init.startswith('he'):
            for l in self.layers:
                torch.nn.init.kaiming_normal_(l.weight.data, nonlinearity=activation)
                if init.startswith('he-x'):
                    l.weight.data.mul_(float(init[4:]))
                if l.bias is not None:
                    l.bias.data.zero_()
        elif init.startswith('sm'):
            for name, param in self.named_parameters():
                param.data.normal_(0, 0.1 ** float(init[2:]))
        else:
            raise NotImplementedError()
    
    def forward(self, x):
        x = self.features(x)
        out = self.final_layer(x)
        if self.dimO == 1:
            out = out.squeeze(dim=1)
        return out


def mlpn(args):
    net = MLPN(L=args.depth, dimD=args.dimD, dimH=args.dimH, dimO=args.dimO, init=args.init, activation=args.activation, first_layer_bias=args.first_layer_bias, norm_layer=lambda w: get_norm1d(w, args))
    return KModel, net
