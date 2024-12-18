import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def difference_loss(pred, target=None):
    pred_last = torch.cat((pred[:, 1:], pred[:, 0][:, None]), 1)
    pred_difference = torch.abs(pred - pred_last) / pred_last
    pred_difference = pred_difference.mean(1)
    return pred_difference


@LOSSES.register_module
class DifferenceLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DifferenceLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight  # 整个损失项的权重

    def forward(self,
                pred,
                target=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss_bbox_difference = self.loss_weight * difference_loss(
            pred,
            None,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)  # weight变量负责一个batch中loss的分配
        return loss_bbox_difference
