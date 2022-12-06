## Based on: https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/preresnet.py
from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math
import klib.si.utils
import klib.train_utils

from klib.block.preresnet import BasicBlock, Bottleneck

from klib.si.kmodel import KModelSIA

__all__ = ['si_preresnet20', 'si_preresnet32', 'si_preresnet56']


class SIPreResNet(nn.Module):
    """Scale-invariant PreResNet
    """

    def __init__(self, depth, num_classes=10, block_name='BasicBlock', final_layer_code='linear-fixed', norm_layer=None):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n, norm_layer)
        self.layer2 = self._make_layer(block, 32, n, norm_layer, stride=2)
        self.layer3 = self._make_layer(block, 64, n, norm_layer, stride=2)
        self.norm4 = norm_layer(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.final_layer = klib.si.utils.get_final_layer(final_layer_code, 64 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        


    def _make_layer(self, block, planes, blocks, norm_layer, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.norm4(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.final_layer(x)
        return x


def _preresnet(depth, args):
    norm_layer = lambda w: klib.train_utils.get_norm2d(w, args, eps=0, affine=False)
    net = SIPreResNet(depth=depth, final_layer_code=args.final_layer_code, norm_layer=norm_layer)
    return KModelSIA, net, set(p for p in net.parameters() if p.requires_grad)


def si_preresnet20(args):
    return _preresnet(20, args)


def si_preresnet32(args):
    return _preresnet(32, args)


def si_preresnet56(args):
    return _preresnet(56, args)
