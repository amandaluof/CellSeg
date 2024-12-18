# Encoding=utf-8

import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms, multiclass_nms_with_mask
from mmdet.ops import ModulatedDeformConvPack

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob, build_norm_layer
from IPython import embed
import cv2
import numpy as np
import math
import time

INF = 1e8


@HEADS.register_module
class PolarMask_Double_GT_Head(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 use_dcn=False,
                 mask_nms=False,
                 loss_cls=dict(type='FocalLoss',
                               use_sigmoid=True,
                               gamma=2.0,
                               alpha=0.25,
                               loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_radius=dict({}),
                 loss_angle=dict({}),
                 loss_centerness=dict(type='CrossEntropyLoss',
                                      use_sigmoid=True,
                                      loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 num_sample_points=360):
        super(PolarMask_Double_GT_Head, self).__init__()
        assert 360 % num_sample_points == 0, '360 is not multiple to sampling interval'

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_mask = build_loss(loss_radius)
        # add by amd
        self.loss_angle = build_loss(loss_angle)

        self.loss_centerness = build_loss(loss_centerness)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        # xez add for polarmask
        self.use_dcn = use_dcn
        self.mask_nms = mask_nms

        # add for less sample points,用num_sample_points个点来作为查询的gt
        self.num_sample_points = num_sample_points

        # debug vis img
        self.vis_num = 1000
        self.count = 0

        # test
        #self.angles = torch.range(0, 350, 10).cuda() / 180 * math.pi

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()  # 预测极径
        self.angle_convs = nn.ModuleList(
        )  # add by amd，polar_angle predicting branch
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if not self.use_dcn:
                self.cls_convs.append(
                    ConvModule(chn,
                               self.feat_channels,
                               3,
                               stride=1,
                               padding=1,
                               conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg,
                               bias=self.norm_cfg is None))
                self.reg_convs.append(
                    ConvModule(chn,
                               self.feat_channels,
                               3,
                               stride=1,
                               padding=1,
                               conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg,
                               bias=self.norm_cfg is None))
                self.mask_convs.append(
                    ConvModule(chn,
                               self.feat_channels,
                               3,
                               stride=1,
                               padding=1,
                               conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg,
                               bias=self.norm_cfg is None))
                self.angle_convs.append(
                    ConvModule(chn,
                               self.feat_channels,
                               3,
                               stride=1,
                               padding=1,
                               conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg,
                               bias=self.norm_cfg is None))
            else:
                self.cls_convs.append(
                    ModulatedDeformConvPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                if self.norm_cfg:
                    self.cls_convs.append(
                        build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.cls_convs.append(nn.ReLU(inplace=True))

                self.reg_convs.append(
                    ModulatedDeformConvPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                if self.norm_cfg:
                    self.reg_convs.append(
                        build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.reg_convs.append(nn.ReLU(inplace=True))

                self.mask_convs.append(
                    ModulatedDeformConvPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                if self.norm_cfg:
                    self.mask_convs.append(
                        build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.mask_convs.append(nn.ReLU(inplace=True))

        self.polar_cls = nn.Conv2d(self.feat_channels,
                                   self.cls_out_channels,
                                   3,
                                   padding=1)  # 如果所有前景类的概率小于0.5，则为背景
        self.polar_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.polar_mask = nn.Conv2d(self.feat_channels, 36, 3, padding=1)
        self.polar_angle = nn.Conv2d(self.feat_channels, 36, 3,
                                     padding=1)  # add by amd, 预测极角(绝对角度)
        self.polar_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales_bbox = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.scales_mask = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        if not self.use_dcn:
            for m in self.cls_convs:
                normal_init(m.conv, std=0.01)
            for m in self.reg_convs:
                normal_init(m.conv, std=0.01)
            for m in self.mask_convs:
                normal_init(m.conv, std=0.01)
            for m in self.angle_convs:
                normal_init(m.conv, std=0.01)  # add by amd
        else:
            pass

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.polar_cls, std=0.01, bias=bias_cls)
        normal_init(self.polar_reg, std=0.01)
        normal_init(self.polar_mask, std=0.01)
        normal_init(self.polar_angle, std=0.01)
        normal_init(self.polar_centerness, std=0.01)

    def forward(self, feats):
        '''
        对传入的多层特征，先分层计算得到若干变量，然后将各层得到的结果按照变量进行合并
        '''
        return multi_apply(self.forward_single, feats, self.scales_bbox,
                           self.scales_mask)

    def forward_single(self, x, scale_bbox, scale_mask):
        '''
        计算三个分支，其中分类分支最后会分别预测分类分数和centerness
        bbox预测分支：预测的是到四条边距离的指数，因为预测始终应该为正
        mask预测分支：预测的是极径的指数，因为预测始终应该为正
        angle预测分支：预测的是极角之差
        '''
        cls_feat = x
        reg_feat = x
        mask_feat = x
        angle_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.polar_cls(cls_feat)  # 只是分数，并未换成概率
        centerness = self.polar_centerness(cls_feat)  # 只是数值

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)  # 逐层运算
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale_bbox(self.polar_reg(reg_feat)).float().exp()

        for mask_layer in self.mask_convs:
            mask_feat = mask_layer(mask_feat)
        mask_pred = scale_mask(self.polar_mask(mask_feat)).float().exp()

        for angle_layer in self.angle_convs:
            angle_feat = angle_layer(angle_feat)
        # angle_pred = self.polar_angle(
        #    angle_feat).float().sigmoid() * 2 * math.pi - math.pi  # 角度在正负pi之间
        angle_pred = self.polar_angle(angle_feat).float()

        # print('angle', angle_pred.reshape(-1, 36).mean(1))
        return cls_score, bbox_pred, centerness, mask_pred, angle_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'mask_preds',
                          'angle_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             mask_preds,
             angle_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_masks,
             gt_bboxes_ignore=None,
             extra_data=None):
        '''
        cls_scores: 所有level的像素的分类预测得分[(batch,80,H1,W1),(batch,80,H2,W2)...(batch,80,H5,W5)]
        bbox_preds: 包含所有level的像素到4条边的预测距离[(batch,4,H1,W1),(batch,4,H2,W2)...(batch,4,H5,W5)]
        centernesses: 包含所有level的像素的预测中心度[(batch,1,H1,W1),(batch,1,H2,W2)...(batch,1,H5,W5)]
        mask_preds: 包含所有level的像素到边界的36个点的预测距离[(batch,36,H1,W1),(batch,36,H2,W2)...(batch,36,H5,W5)]
        angle_preds: 包含所有level的像素的36条射线的预测角度[(batch,36,H1,W1),(batch,36,H2,W2)...(batch,36,H5,W5)]
        gt_bboxes: 未使用，[tensor(num_bbox,4),...,tensor(num_bbox,4)],list长度为batchSize,
        gt_labels: 未使用
        gt_mask: 未使用
        extra_data：包含键：_gt_labels,_gt_bboxes,_gt_masks,_contour_all_points，包含所有特征像素的标签、到bbox四条边的距离、等间隔的36个点的极径、360个点的极径
        '''
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(
            mask_preds) == len(angle_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(
            featmap_sizes, bbox_preds[0].dtype,
            bbox_preds[0].device)  # list，长度为特征的层数，所有特征点的坐标

        # target
        labels, bbox_targets, mask_targets, angle_targets = self.polar_target(
            all_level_points, extra_data)

        # predict
        num_imgs = cls_scores[0].size(0)  # batch size
        # flatten cls_scores, bbox_preds and centerness,最终得到长度为batch的list[(num_pixel,80)...(num_pixel,80)]
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]  # 放入了batch size的维度，[(num_pixel,80),(num_pixel,80)]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_mask_preds = [
            mask_pred.permute(0, 2, 3, 1).reshape(-1, 36)
            for mask_pred in mask_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 36)
            for angle_pred in angle_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)  # [num_pixel, 80]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)  # [num_pixel, 4]
        flatten_mask_preds = torch.cat(flatten_mask_preds)  # [num_pixel, 36]
        flatten_angle_preds = torch.cat(flatten_angle_preds)  # [num_pixel, 36]
        flatten_centerness = torch.cat(flatten_centerness)  # [num_pixel]

        # target
        flatten_labels = torch.cat(labels).long()  # [num_pixel]
        flatten_bbox_targets = torch.cat(bbox_targets)  # [num_pixel, 4]
        flatten_mask_targets = torch.cat(mask_targets)  # [num_pixel, 36]
        flatten_angle_targets = torch.cat(angle_targets)  # [num_pixel, 36]
        flatten_points = torch.cat([
            points.repeat(num_imgs, 1) for points in all_level_points
        ])  # [batch*num_pixel,2] 放入batch size的维度

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)

        loss_dict = {}
        loss_cls = self.loss_cls(flatten_cls_scores,
                                 flatten_labels,
                                 avg_factor=num_pos +
                                 num_imgs)  # avoid num_pos is 0
        loss_dict['loss_cls'] = loss_cls
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]  # 预测的centerness
        pos_mask_preds = flatten_mask_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_mask_targets = flatten_mask_targets[pos_inds]
            pos_angle_targets = flatten_angle_targets[pos_inds]
            pos_centerness_targets = self.polar_centerness_target(
                flatten_mask_targets[pos_inds])

            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)

            # centerness weighted iou loss,训练阶段用centerness真值加权，验证阶段用预测的centerness
            loss_bbox = self.loss_bbox(pos_decoded_bbox_preds,
                                       pos_decoded_target_preds,
                                       weight=pos_centerness_targets,
                                       avg_factor=pos_centerness_targets.sum())
            loss_dict['loss_bbox'] = loss_bbox

            # 极径损失
            loss_mask = self.loss_mask(pos_mask_preds,
                                       pos_mask_targets,
                                       weight=pos_centerness_targets,
                                       avg_factor=pos_centerness_targets.sum())
            loss_dict['loss_radius'] = loss_mask

            # 极角损失
            loss_angle = self.loss_angle(
                pos_angle_preds,
                pos_angle_targets,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
            loss_dict['loss_angle'] = loss_angle

            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
            loss_dict['loss_centerness'] = loss_centerness

        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_mask = pos_mask_preds.sum()
            loss_angle = pos_angle_preds.sum()
            loss_centerness = pos_centerness.sum()
            loss_dict['loss_bbox'] = loss_bbox
            loss_dict['loss_radius'] = loss_mask
            loss_dict['loss_centerness'] = loss_centerness
            loss_dict['loss_angle'] = loss_angle

        loss_info = ''
        for k, v in loss_dict.items():
            loss_info += '{}:{}, '.format(k, v)
        loss_info = loss_info[:-1]

        return loss_dict

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(0,
                               w * stride,
                               stride,
                               dtype=dtype,
                               device=device)
        y_range = torch.arange(0,
                               h * stride,
                               stride,
                               dtype=dtype,
                               device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2  # 坐标
        return points

    def polar_target(self, points, extra_data):
        '''
        extra_data:{_gt_labels:,_gt_bboxes:,_gt_masks:,_contour_all_points:}
        返回的_contour_all_points是360个点对应的极径
        '''
        assert len(points) == len(self.regress_ranges)

        num_levels = len(points)

        labels_list, bbox_targets_list, mask_targets_list, angle_targets_list = extra_data.values(
        )
        # split to per img, per level (每张图是[15360, 3840, 960, 240, 60])
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0)
                       for labels in labels_list]  # shape:(2,5,num_pixels)
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        mask_targets_list = [
            mask_targets.split(num_points, 0)
            for mask_targets in mask_targets_list
        ]  # 尺寸:2,5,像素个数,36
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]  # 尺寸:2,5,像素个数,36

        # concat per level image,将不同图相同level的gt concat
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_mask_targets = []
        concat_lvl_angle_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
            concat_lvl_mask_targets.append(
                torch.cat(
                    [mask_targets[i] for mask_targets in mask_targets_list]))
            concat_lvl_angle_targets.append(
                torch.cat([
                    angle_targets[i] for angle_targets in angle_targets_list
                ]))

        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_mask_targets, concat_lvl_angle_targets

    def polar_centerness_target(self, pos_mask_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        centerness_targets = (pos_mask_targets.min(dim=-1)[0] /
                              pos_mask_targets.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   mask_preds,
                   angle_preds,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            mask_pred_list = [
                mask_preds[i][img_id].detach() for i in range(num_levels)
            ]
            angle_pred_list = [
                angle_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(
                cls_score_list, bbox_pred_list, mask_pred_list,
                angle_pred_list, centerness_pred_list, mlvl_points, img_shape,
                scale_factor, cfg, rescale
            )  # tuple(det_bboxes, det_labels, det_masks) # (center,radius,angles)
            result_list.append(
                det_bboxes)  # [(det_bboxes, det_labels, det_masks),()...]
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mask_preds,
                          angle_preds,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        '''
        :cls_scores: [tensor(80,H1,W1),tensor,...tensor]长度为5的数组，每个tensor表示对应尺寸的各个点对应的预测结果
        :bbox_preds: [tensor(4,H1,W1),tensor,...tensor]长度为5的数组，每个tensor表示对应尺寸的各个点对应的预测结果
        :mask_preds: [tensor(36,H1,W1),tensor,...tensor]长度为5的数组，每个tensor表示对应尺寸的各个点对应的预测结果
        '''
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_masks = []
        mlvl_angles = []
        mlvl_centerness = []
        # add by amd
        mlvl_center = []
        mlvl_radius = []
        mlvl_angle = []

        # 遍历每个level的预测结果
        for cls_score, bbox_pred, mask_pred, angle_pred, centerness, points in zip(
                cls_scores, bbox_preds, mask_preds, angle_preds, centernesses,
                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            mask_pred = mask_pred.permute(1, 2, 0).reshape(-1, 36)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, 36)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                mask_pred = mask_pred[topk_inds, :]
                angle_pred = angle_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            # masks = distance2mask(
            #     points, mask_pred, self.angles, max_shape=img_shape)

            masks = distance2mask(points,
                                  mask_pred,
                                  angle_pred,
                                  max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_masks.append(masks)
            # add by amd
            mlvl_center.append(points)
            mlvl_radius.append(mask_pred)
            mlvl_angle.append(angle_pred)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_masks = torch.cat(mlvl_masks)
        # add by amd
        mlvl_center = torch.cat(mlvl_center)  #(3260,2)
        mlvl_radius = torch.cat(mlvl_radius)  # (3260,36)
        mlvl_angle = torch.cat(mlvl_angle)  # (3260,36)
        if rescale:
            _mlvl_bboxes = mlvl_bboxes / mlvl_bboxes.new_tensor(scale_factor)
            try:
                scale_factor = torch.Tensor(scale_factor)[:2].cuda().unsqueeze(
                    1).repeat(1, 36)
                _mlvl_masks = mlvl_masks / scale_factor
                ###
                scale_factor_center = scale_factor[:, 0].unsqueeze(0)
                _mlvl_center = mlvl_center / scale_factor_center
                _mlvl_radius = mlvl_radius / scale_factor_center.min(
                )  # approxiamte for ridus, greater than real
                ###
            except:
                _mlvl_masks = mlvl_masks / mlvl_masks.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        # mask centerness is smaller than origin centerness, so add a constant is important or the score will be too low.
        centerness_factor = 0.5
        if self.mask_nms:
            '''1 mask->min_bbox->nms, performance same to origin box'''
            a = _mlvl_masks
            _mlvl_bboxes = torch.stack([
                a[:, 0].min(1)[0], a[:, 1].min(1)[0], a[:, 0].max(1)[0],
                a[:, 1].max(1)[0]
            ], -1)
            det_bboxes, det_labels, det_masks = multiclass_nms_with_mask(
                _mlvl_bboxes,
                mlvl_scores,
                _mlvl_masks,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_centerness + centerness_factor)

        # original
        # else:
        #     '''2 origin bbox->nms, performance same to mask->min_bbox'''
        #     det_bboxes, det_labels, det_masks = multiclass_nms_with_mask(
        #         _mlvl_bboxes,
        #         mlvl_scores,
        #         _mlvl_masks,
        #         cfg.score_thr,
        #         cfg.nms,
        #         cfg.max_per_img,
        #         score_factors=mlvl_centerness + centerness_factor)

        # return det_bboxes, det_labels, det_masks
        else:
            '''2 origin bbox->nms, performance same to mask->min_bbox'''
            vis_info = {
                'center': _mlvl_center,
                'radius': _mlvl_radius,
                'angle': mlvl_angle
            }
            det_bboxes, det_labels, det_masks, det_centers, det_radius, det_angles = multiclass_nms_with_mask(
                _mlvl_bboxes,
                mlvl_scores,
                _mlvl_masks,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_centerness + centerness_factor,
                vis_info=vis_info)
        return det_bboxes, det_labels, det_masks, det_centers, det_radius, det_angles


# test
# not calculate accumlative angle, only for polarmask_double_gt_head
def distance2mask(points, distances, angles, max_shape=None):
    '''Decode distance prediction to 36 mask points
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 36,from angle 0 to 350.
        angles (Tensor): 预测的delta theta
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded masks.
    '''
    num_points = points.shape[0]
    points = points[:, :, None].repeat(1, 1, 36)
    c_x, c_y = points[:, 0], points[:, 1]

    sin = torch.sin(angles)
    cos = torch.cos(angles)

    #distances = distances.mean(1)[:, None].repeat(1, 36)

    x = distances * sin + c_x
    y = distances * cos + c_y

    if max_shape is not None:
        x = x.clamp(min=0, max=max_shape[1] - 1)
        y = y.clamp(min=0, max=max_shape[0] - 1)

    res = torch.cat([x[:, None, :], y[:, None, :]], dim=1)
    return res
