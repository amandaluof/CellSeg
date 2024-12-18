import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module
class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss
    """
    def __init__(self,
                 gamma=2,
                 alpha=0.25,
                 use_sigmoid=True,
                 reduction='sum',
                 loss_weight=1.0):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, _input, target, avg_factor):
        if self.use_sigmoid:
            pt = torch.sigmoid(_input).clamp(1e-12, 1e12)  # 防止loss爆炸
        else:
            NotImplementedError
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
            (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            # if torch.isnan(torch.mean(loss)):
            #     import pdb
            #     pdb.set_trace()  # 212开始，因为log p，因真数为0，因此loss为nan
            loss = self.loss_weight * torch.mean(loss)
        elif self.reduction == 'sum':
            loss = self.loss_weight * torch.sum(loss) / avg_factor
        return loss
