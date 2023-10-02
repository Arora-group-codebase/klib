import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class AccuracySum(nn.Module):
    def __init__(self):
        super(AccuracySum, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target).to(torch.get_default_dtype()).sum()


def separate_logits(y, target) -> Tuple[torch.Tensor, torch.Tensor]:
    N = y.shape[0]
    C = y.shape[1]
    correct = y[range(N), target]
    wrong = torch.masked_select(y, torch.logical_not(F.one_hot(target, num_classes=C))).reshape([N, C - 1])
    return correct, wrong
    

class MultiLabelMargin(nn.Module):
    def __init__(self):
        super(MultiLabelMargin, self).__init__()

    def forward(self, input, target):
        correct, wrong = separate_logits(input, target)
        return (correct - wrong.max(dim=1).values).min()


class LogCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(LogCrossEntropyLoss, self).__init__()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        c, w = separate_logits(input, target)
        diff = (w - c.view([-1, 1])).double()
        if diff.max() >= -30:
            return diff.exp().sum(dim=1).log1p().sum().log().to(input.dtype)
        else:
            return diff.logsumexp(dim=[0, 1]).to(input.dtype)


class CELNR(nn.Module): # LNR = label noise regularization
    def __init__(self, label_smoothing) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ls = label_smoothing
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        C = input.shape[1]
        random_target = torch.randint_like(target, C)
        mask = (torch.rand_like(target, dtype=torch.get_default_dtype()) < self.ls).to(target.dtype)
        target = target * (1 - mask) + random_target * mask
        return self.ce(input, target)


@torch.no_grad()
def count_correct(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k)
    return res


@torch.no_grad()
def binary_count_correct(output, target):
    """
    Args:
        output: logits
        target: 0-1 labels
    """
    return ((output > 0) * target + (output <= 0) * (1 - target)).sum().float()