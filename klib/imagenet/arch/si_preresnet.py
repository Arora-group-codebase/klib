## Based on: https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/preresnet.py
## https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet50
from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/facebookarchive/fb.resnet.torch/blob/master/models/preresnet.lua
(c) YANG, Wei
'''
import torch.nn as nn
import math
import klib.si.utils
import klib.train_utils

from klib.block.preresnet import BasicBlock, Bottleneck

from klib.si.kmodel import KModelSIA

__all__ = ['si_preresnet50', 'si_preresnet101', 'si_preresnet152']


class SIPreResNet(nn.Module):
    """Scale-invariant PreResNet
    """

    def __init__(self, layers, num_classes=1000, block_name='BasicBlock', final_layer_code='linear-fixed', norm_layer=None):
        super().__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], norm_layer, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], norm_layer, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], norm_layer, stride=2)
        self.norm5 = norm_layer(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_layer = klib.si.utils.get_final_layer(final_layer_code, 512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        

    def _make_layer(self, block, planes, blocks, norm_layer, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                norm_layer(self.inplanes),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.norm5(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.final_layer(x)
        return x


def _preresnet(layers, args):
    norm_layer = lambda w: klib.train_utils.get_norm2d(w, args, eps=0, affine=False)
    net = SIPreResNet(layers=layers, block_name='bottleneck', final_layer_code=args.final_layer_code, norm_layer=norm_layer)
    return KModelSIA, net, set(p for p in net.parameters() if p.requires_grad)


def si_preresnet50(args):
    return _preresnet([3, 4, 6, 3], args)


def si_preresnet101(args):
    return _preresnet([3, 4, 23, 3], args)


def si_preresnet152(args):
    return _preresnet([3, 8, 36, 3], args)
