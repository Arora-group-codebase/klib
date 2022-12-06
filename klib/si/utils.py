import math
import torch
from torch import nn
from torch.nn import LayerNorm


def get_final_layer(code: str, in_features: int, num_classes: int = 1000):
    subcode = code.split('+')
    
    fc = nn.Linear(in_features, num_classes)
    
    if subcode[0] == 'linear-fixed':
        fc.weight.data.normal_(0, math.sqrt(1 / fc.in_features))
        fc.bias.data.zero_()
    else:
        assert subcode[0] == 'linear-fixed-etf'

        with torch.no_grad():
            nn.init.orthogonal_(fc.weight.data)
            fc.weight.data = (math.sqrt(num_classes / (num_classes - 1)) * (torch.eye(num_classes) - torch.ones((num_classes, num_classes)) / num_classes)) @ fc.weight.data
        
        fc.bias.data.zero_()

    if len(subcode) == 1:
        fc.weight.requires_grad = False
        fc.bias.requires_grad = False
        return fc
    
    assert len(subcode) == 2
    
    m = LayerNorm([num_classes], eps=0, elementwise_affine=True)
    m.weight.requires_grad = False
    m.bias.requires_grad = False

    for i in range(1, 6):
        if subcode[1] == f'ln-fixed-{i}':
            m.weight.data.fill_(i)
            return nn.Sequential(fc, m)
    
    assert subcode[1] == 'ln-fixed-ls-0.1' # optimum is equivalent to label smoothing with eps=0.1
    eps = 0.1
    m.weight.data.fill_((num_classes - 1) ** 0.5 / num_classes * math.log(1 + num_classes * (1 / eps - 1)))
    return nn.Sequential(fc, m)
