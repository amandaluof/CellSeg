import torch.nn as nn
import torch.nn.functional as F

from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss
from IPython import embed
import torch
import math


@LOSSES.register_module
class MaskIOULoss(nn.Module):
    def __init__(self, deviation=False):
        super(MaskIOULoss, self).__init__()
        self.deviation = deviation

    def forward(self,
                pred_angle,
                pred_radius,
                target_radius,
                weight,
                avg_factor=None):
        '''
        IOU = sum(d_min^2*theta)/sum(d_max^2*theta)
        按照点的中心度进行加权
        '''
        total = torch.stack([pred_radius, target_radius], -1)  # 维度变为(N,36,2)
        l_max = total.max(dim=2)[0].pow(2)  # 会同时返回值和下标，因此取第一个元素
        l_min = total.min(dim=2)[0].pow(2).clamp(min=1e-6)
        # theta_gradient_numerator = l_max * (
        #     (l_min * pred_angle).sum(axis=1, keepdim=True)) - l_min * (
        #         (l_max * pred_angle).sum(axis=1, keepdim=True))
        # theta_gradient_denominator = (
        #     (l_min * pred_angle).sum(axis=1, keepdim=True)) * (
        #         (l_max * pred_angle).sum(axis=1, keepdim=True))
        # theta_gradient = (theta_gradient_numerator /
        #                   theta_gradient_denominator / 4).mean(0)

        # gradient_radius = torch.zeros_like(pred_radius,
        #                                    requires_grad=False).cuda()
        # gradient_radius = torch.where(
        #     pred_radius > target_radius, 2 * pred_angle * pred_radius /
        #     (l_max * pred_angle).sum(axis=1, keepdim=True), 2 * pred_angle *
        #     pred_radius / (l_min * pred_angle).sum(axis=1, keepdim=True))

        # print('**********************************')
        # print('pred_angle:{}'.format(torch.cumsum(pred_angle, 1).data))
        # print('gradient_angle:{}'.format(theta_gradient))
        # print('pred_radius:{}'.format(pred_radius))
        # print('target_radius:{}'.format(target_radius))
        # print('gradient_radius:{}'.format(gradient_radius))

        # check gradient
        # a = l_max / (l_max * pred_angle).sum(axis=1, keepdim=True)  # (n,36)
        # b = l_min / (l_min * pred_angle).sum(axis=1, keepdim=True)  # (n,36)
        # print('angle gradient', (a - b).mean(0).mean())

        if self.deviation:
            # input pred_angle is deviation of delta theta
            angle_init = (math.pi / 180 * 10 *
                          torch.ones(pred_angle.shape)).cuda()
            pred_angle = angle_init + pred_angle

        loss = ((pred_angle * l_max).sum(dim=1) /
                (pred_angle * l_min).sum(dim=1)).log()

        loss = loss * weight
        loss = loss.sum() / avg_factor  # calcute avg loss

        return loss


@LOSSES.register_module
class MaskIOULoss_v0(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MaskIOULoss_v0, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, avg_factor=None):
        '''
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         '''

        total = torch.stack([pred, target], -1)  # 维度变为(N,36,2)
        l_max = total.max(dim=2)[0]  # 会同时返回值和下标，因此取第一个元素
        l_min = total.min(dim=2)[0]
        loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = loss * weight
        loss = loss.sum() / avg_factor
        loss = loss * self.loss_weight
        return loss


@LOSSES.register_module
class MaskIOULoss_v2(nn.Module):
    def __init__(self):
        super(MaskIOULoss_v2, self).__init__()

    def forward(self, pred, target, weight, avg_factor=None):
        '''
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         '''

        total = torch.stack([pred, target], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0].clamp(min=1e-6)

        # loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = (l_max / l_min).log().mean(dim=1)
        loss = loss * weight
        loss = loss.sum() / avg_factor
        return loss


@LOSSES.register_module
class MaskIOULoss_v3(nn.Module):
    def __init__(self):
        super(MaskIOULoss_v3, self).__init__()

    def forward(self, pred, target, weight, avg_factor=None):
        '''
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         '''

        total = torch.stack([pred, target], -1)
        l_max = total.max(dim=2)[0].pow(2)
        l_min = total.min(dim=2)[0].pow(2)

        # loss = 2 * (l_max.prod(dim=1) / l_min.prod(dim=1)).log()
        # loss = 2 * (l_max.log().sum(dim=1) - l_min.log().sum(dim=1))
        loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = loss * weight
        loss = loss.sum() / avg_factor
        return loss
