'''
Author: your name
Date: 2021-11-08 11:17:38
LastEditTime: 2021-12-26 15:02:00
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /PolarMask/mmdet/models/losses/l2_loss.py
'''
import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def l2_loss(pred, target, beta=1.0):
    '''
    pred: shape:(num_pixel, 36，2) / (num_pixel, 4)
    '''
    assert pred.size() == target.size() and target.numel() > 0
    if len(pred.shape) == 2:
        pred = pred.reshape(pred.shape[0], -1, 2)  # (num_pixel,2,2)
        target = target.reshape(target.shape[0], -1, 2)

    loss = torch.sqrt(
        ((pred - target)**2).sum(2)).sum(1) / pred.shape[1]  # 36个点误差之和 / 左上、右下2个点的误差之和

    return loss


@LOSSES.register_module
class L2Loss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(L2Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                normalize_factor=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        assert pred.shape == target.shape
        if normalize_factor is not None:
            target = target * normalize_factor
        loss_bbox = self.loss_weight * l2_loss(pred,
                                               target,
                                               weight,
                                               beta=self.beta,
                                               reduction=reduction,
                                               avg_factor=avg_factor,
                                               **kwargs)

        return loss_bbox
