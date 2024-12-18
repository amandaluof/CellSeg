import torch
import torch.nn as nn
from mmcv.cnn import normal_init, constant_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms, multiclass_nms_with_mask
from mmdet.ops import ModulatedDeformConvPack
import torch.nn.functional as F

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, NonLocalModule, SnakeBlock, SnakeProBlock, FuseIntermediateBlock, Scale, Scale_list, bias_init_with_prob, build_norm_layer
from IPython import embed
import cv2
import numpy as np
import math
import time

INF = 1e8


@HEADS.register_module
class PolarMask_Refine_Head(nn.Module):
    def __init__(
            self,
            num_classes,
            in_channels,
            feat_channels=256,
            stacked_convs=4,
            stacked_init_convs=4,
            stacked_refine_convs=0,
            cascade_refine_num=0,
            stacked_additinal_cls_convs=0,
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
            loss_mask_init=dict({}),
            loss_mask_refine=dict({}),
            loss_pts_bbox_init=dict({}),
            loss_pts_bbox_refine=dict({}),
            loss_heatmap=dict({}),
            loss_centerness=dict(type='CrossEntropyLoss',
                                 use_sigmoid=True,
                                 loss_weight=1.0),
            loss_refine_weight=[],
            conv_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            num_sample_points=36,
            normalize_factor=1,
            gradient_mul=0.1,
            sample_last=False,
            non_local=False,
            refine_mask=True,
            refine_seperate=True,
            refine_1d=False,
            additional_cls_branch=False,
            additional_cls_single=True,
            all_instances_heatmap=True,
            polar_xy=False,
            predict_refine_first=False,
            heatmap_upsample=False,
            heatmap_upsample_deconv=False,
            gt_unormalize=False,
            polar_xy_represent=False,
            polar_both=False,
            polarxy_polarxy=False,
            centerness_base='boundary',
            fuse_refine_feat={
                'center': None,
                'coord': None,
                'adj': None,
                'snake': None,
                'global': None,
                'intermediate': None,
            },
            intermediate_feat_level=[],
            coord_radius=False):
        super(PolarMask_Refine_Head, self).__init__()
        assert 360 % num_sample_points == 0, '360 is not multiple to sampling interval'
        if cascade_refine_num > 0:
            assert loss_mask_refine[
                'loss_weight'] == 1, 'use loss_refine_weight to set weights of loss in refine stage'
            assert len(
                loss_refine_weight) == cascade_refine_num + 1  # 原有精修+级联精修的权重
        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.stacked_init_convs = stacked_init_convs
        self.stacked_refine_convs = stacked_refine_convs
        self.stacked_additinal_cls_convs = stacked_additinal_cls_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        # add by amd
        self.loss_mask_init = build_loss(loss_mask_init)
        self.loss_mask_refine = build_loss(loss_mask_refine)
        if len(loss_heatmap) > 0:
            self.loss_heatmap = build_loss(loss_heatmap)
        self.loss_pts_bbox_init = build_loss(
            loss_pts_bbox_init) if len(loss_pts_bbox_init) > 0 else None
        self.loss_pts_bbox_refine = build_loss(
            loss_pts_bbox_refine) if len(loss_pts_bbox_refine) > 0 else None
        self.loss_centerness = build_loss(loss_centerness)
        self.normalize_factor = normalize_factor
        self.gradient_mul = gradient_mul
        self.sample_last = sample_last  # 是否在更高分辨率的feature map上进行采样
        self.non_local = non_local
        self.refine_mask = refine_mask  # 在验证时，使用精修的mask还是初始mask
        self.refine_seperate = refine_seperate  # 在精修阶段，各点的偏移是否独立回归
        self.additional_cls_branch = additional_cls_branch  # 是否使用额外的分类分支，用于正则/特征融合等
        self.additional_cls_single = additional_cls_single  # 是否对P3层进行正则形式的监督
        self.all_instances_heatmap = all_instances_heatmap  # 是否利用所有层的实例缩绘制到P3进行监督
        self.polar_xy = polar_xy  # new: 第一阶段是否预测是的polar对应的gt # 精修的方式是否是先预测36个极径，再精修
        self.predict_refine_first = predict_refine_first  # 是否先预测精修偏移再采样
        self.cascade_refine_num = cascade_refine_num  # 在1个初始偏移、1个精修偏移后的精修偏移预测次数
        self.loss_refine_weight = loss_refine_weight  # 级联精调时的损失权重
        self.heatmap_upsample = heatmap_upsample  # 是否将heatmap分支上采样到1/4原图
        self.heatmap_upsample_deconv = heatmap_upsample_deconv  # 在对heatmap上采样时是否使用转置卷积
        self.gt_unormalize = gt_unormalize  # 是否对使用的gt进行反归一化，True:原始值，False：归一化后的值
        self.polar_xy_represent = polar_xy_represent  # 在使用polarmask的gt时是否使用二维坐标
        self.polar_both = polar_both  # 是否 两个阶段同时预测极径
        self.polarxy_polarxy = polarxy_polarxy  # 是否第二阶段预测的也是polar的二维偏移形式,若为否，则第二阶段预测的等间隔gt
        self.centerness_base = centerness_base  # 计算centerness gt时使用的点是轮廓上的点或是用polar极径计算(与原论文一致)
        self.fuse_center_feat = fuse_refine_feat[
            'center']  # refine阶段是否对各特征进行融合
        self.fuse_coord_feat = fuse_refine_feat['coord']
        self.coord_radius = coord_radius  # fuse坐标信息时concat极径
        self.fuse_adj_feat = fuse_refine_feat['adj']
        self.fuse_snake_feat = fuse_refine_feat['snake']
        self.fuse_global_feat = fuse_refine_feat['global']
        self.fuse_intermediate_feat = fuse_refine_feat['intermediate']
        if self.fuse_intermediate_feat:
            self.intermediate_feat_level = intermediate_feat_level
        self.refine_1d = refine_1d  # 是否36个点共享相同的卷积

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

        self.angles = torch.range(0, 350, 10).cuda() / 180 * math.pi

        self._init_layers()

        if self.heatmap_upsample:
            print('use heatmap with 1/4 original size')
        if self.heatmap_upsample_deconv:
            print('use deconv to upsample the P3 for heatmap gt making')
        if self.gt_unormalize:
            print('gt deviation should not be normalized !!!!')
        if self.polar_both:
            assert self.polar_xy == True
            print('both predict polar radius in two stages')
        if self.polarxy_polarxy:
            assert self.polar_xy == True
            print('both predict xy-fromat of polar gt points')

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.coordinate_convs = nn.ModuleList(
        )  # 预测轮廓点坐标，add by amd，boundary point predicting branch
        self.refine_convs = nn.ModuleList()  # 预测精细偏移
        if self.additional_cls_branch:
            self.additional_cls_convs = nn.ModuleList()  # 额外的分类分支

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
        # mask分支的模型
        ## 粗定位
        for i in range(self.stacked_init_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.coordinate_convs.append(
                ConvModule(chn,
                           self.feat_channels,
                           3,
                           stride=1,
                           padding=1,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg,
                           bias=self.norm_cfg is None))


            # if self.non_local:
            #     if i % 2 == 1:
            #         self.coordinate_convs.append(
            #             NonLocalModule(self.feat_channels,
            #                            self.feat_channels,
            #                            stride=1,
            #                            padding=0,
            #                            conv_cfg=self.conv_cfg,
            #                            norm_cfg=self.norm_cfg,
            #                            bias=self.norm_cfg is None)
            #         )  # Non-local中的第1层的bias为True，因此此时无归一化层；第2层的bias为False,因为在第二层后有归一化层



        ## 精修偏移
        if self.refine_seperate:
            if not self.predict_refine_first:
                # 如果使用了coord conv但后面没有跟其他feat 融合，则通道数多2
                if self.fuse_coord_feat and (not self.fuse_adj_feat) and (
                        not self.fuse_global_feat):
                    if not self.coord_radius:
                        tmp_in_channel = self.feat_channels + 2
                    else:
                        tmp_in_channel = self.feat_channels + 1

                else:
                    tmp_in_channel = self.feat_channels

                for i in range(self.stacked_refine_convs):
                    chn = tmp_in_channel if i == 0 else 128
                    self.refine_convs.append(
                        ConvModule(chn * 36,
                                   128 * 36,
                                   1,
                                   stride=1,
                                   padding=0,
                                   groups=36,
                                   conv_cfg=self.conv_cfg,
                                   norm_cfg=dict(type='GN',
                                                 num_groups=36,
                                                 requires_grad=True),
                                   bias=self.norm_cfg is None))
                chn = tmp_in_channel if self.stacked_refine_convs == 0 else 128
                if not self.polar_both:
                    self.refine = nn.Conv2d(
                        chn * 36, 72, 1, padding=0,
                        groups=36)  # 精细偏移预测，分组卷积，各关键点的偏移互不影响
                else:
                    self.refine = nn.Conv2d(
                        chn * 36, 36, 1, padding=0,
                        groups=36)  # 精细偏移预测极径偏移，分组卷积，各关键点的偏移互不影响

            else:
                # 精细偏移预测，在先预测后采样的情况下，不需要分组卷积
                for i in range(self.stacked_refine_convs):
                    chn = self.in_channels if i == 0 else self.feat_channels
                    self.refine_convs.append(
                        ConvModule(chn,
                                   self.feat_channels,
                                   3,
                                   stride=1,
                                   padding=1,
                                   conv_cfg=self.conv_cfg,
                                   norm_cfg=self.norm_cfg,
                                   bias=self.norm_cfg is None))
                self.refine = nn.Conv2d(self.feat_channels, 72, 3, padding=1)
        else:
            chn = self.feat_channels if self.stacked_refine_convs == 0 else 128
            if not self.refine_1d:
                self.refine = nn.Conv2d(chn * 36, 36, 1,
                                        padding=0)  # 使用1x1卷积，但36个点之间协同预测
            else:
                self.refine = nn.Conv1d(chn, 1, 1, padding=0)  # 1维卷积

        if self.polar_both:
            self.relu = nn.ReLU(inplace=True)

        if self.additional_cls_branch:
            if self.heatmap_upsample_deconv:
                self.additional_cls_convs.append(
                    nn.ConvTranspose2d(self.in_channels,
                                       self.in_channels,
                                       2,
                                       stride=2,
                                       padding=0))
            in_chn = [self.in_channels, 128, 64, 64]
            out_chn = [128, 64, 64, 64]
            for i in range(self.stacked_additinal_cls_convs):
                self.additional_cls_convs.append(
                    ConvModule(in_chn[i],
                               out_chn[i],
                               3,
                               stride=1,
                               padding=1,
                               conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg,
                               bias=self.norm_cfg is None))
                # 制作heatmap gt时使用转置卷积

            chn = out_chn[-1]
            self.additional_cls = nn.Conv2d(chn, 1, 3, padding=1)

        self.polar_cls = nn.Conv2d(self.feat_channels,
                                   self.cls_out_channels,
                                   3,
                                   padding=1)  # 如果所有前景类的概率小于0.5，则为背景
        self.polar_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        if self.polar_xy:
            if not self.polar_xy_represent:
                # 第一阶段预测极径
                self.coordinate = nn.Conv2d(self.feat_channels,
                                            36,
                                            3,
                                            padding=1)
            else:
                # 第一阶段预测polar对应的二维偏移
                self.coordinate = nn.Conv2d(self.feat_channels,
                                            72,
                                            3,
                                            padding=1)
        else:
            self.coordinate = nn.Conv2d(self.feat_channels, 72, 3, padding=1)

        # 级联多个精修层，需要多个scale来对每一层预测的偏移进行缩放
        if self.cascade_refine_num > 0:
            self.cascade_refine_convs = nn.ModuleList()
            self.cascade_refine_layers = nn.ModuleList()
            for i in range(self.cascade_refine_num):
                for j in range(self.stacked_refine_convs):
                    chn = self.feat_channels if j == 0 else 128
                    self.cascade_refine_convs.append(
                        ConvModule(chn * 36,
                                   128 * 36,
                                   1,
                                   stride=1,
                                   padding=0,
                                   groups=36,
                                   conv_cfg=self.conv_cfg,
                                   norm_cfg=dict(type='GN',
                                                 num_groups=36,
                                                 requires_grad=True),
                                   bias=self.norm_cfg is None))
                chn = self.feat_channels if self.stacked_refine_convs == 0 else 128
                self.cascade_refine_layers.append(
                    nn.Conv2d(chn * 36, 72, 1, padding=0,
                              groups=36))  # 从backbone中提取的特征进行的单层精修，通道数为256
            assert self.cascade_refine_num == 1, 'Scale module list is not compatible'
            self.cascade_refine_scale = nn.ModuleList(
                [Scale_list([1.0, 1.0]) for _ in self.strides])

        self.polar_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        # feature fusion in refine stage

        # 融合各轮廓点与中心点的特征, 逐点的1d卷积
        if self.fuse_center_feat:
            self.fuse_center = nn.Conv1d(self.feat_channels * 2,
                                         self.feat_channels, 1)

        # 融合相邻点的特征(后面可改为一个list)
        if self.fuse_adj_feat:
            if self.fuse_coord_feat:
                if not self.coord_radius:
                    tmp_in_channel = self.feat_channels + 2
                else:
                    tmp_in_channel = self.feat_channels + 1
            else:
                tmp_in_channel = self.feat_channels
            self.fuse_adj = SnakeBlock(state_dim=tmp_in_channel,
                                       out_state_dim=self.feat_channels,
                                       conv_type='grid',
                                       n_adj=1)

        # 使用deep snake中的多层卷积并使用残差连接和融合全局特征
        if self.fuse_snake_feat:
            self.fuse_snake = SnakeProBlock(state_dim=self.feat_channels,
                                            out_state_dim=self.feat_channels,
                                            feature_dim=128,
                                            res_layer_num=4,
                                            conv_type='grid',
                                            n_adj=1)  # 将每个点的特征压缩为128维

        # 融合使用mask_head中的中间特征用于refine的偏移预测
        if self.fuse_intermediate_feat:
            self.fuse_intermediate = FuseIntermediateBlock(
                len_state=len(self.intermediate_feat_level) + 1,
                fusion_out_state_dim=512)  # fusion_out_state_dim表示全局feat的维度

        # 对每个轮廓点的特征融合所在轮廓的全局特征
        if self.fuse_global_feat:
            if self.fuse_coord_feat and (not self.fuse_adj_feat):
                if not self.coord_radius:
                    tmp_in_channel = (self.feat_channels +
                                      2) * 2  # 融合了坐标但又未经过临近点的特征融合
                else:
                    tmp_in_channel = (self.feat_channels +
                                      1) * 2  # 融合了坐标但又未经过临近点的特征融合
            else:
                tmp_in_channel = self.feat_channels * 2
            self.fuse_global = nn.Conv1d(tmp_in_channel, self.feat_channels, 1)

        self.scales_bbox = nn.ModuleList([Scale(1.0) for _ in self.strides])

        if self.polar_xy:
            if not self.polar_xy_represent:
                self.scales_mask = nn.ModuleList(
                    [Scale(1.0) for _ in self.strides])  # 对每一层的极径进行放缩
            else:
                self.scales_mask = nn.ModuleList(
                    [Scale_list([1.0, 1.0]) for _ in self.strides])
        else:
            self.scales_mask = nn.ModuleList([
                Scale_list([1.0, 1.0]) for _ in self.strides
            ])  # 粗略回归阶段对于xy坐标，应该是不同的scale参数

        if not self.polar_both:
            self.scale_refine = nn.ModuleList([
                Scale_list([1.0, 1.0]) for _ in self.strides
            ])  # 对精修偏移再不同尺度上具有不同的缩放因子
        else:
            self.scale_refine = nn.ModuleList(
                [Scale(1.0) for _ in self.strides])  # 对精修极径在不同尺度上具有不同的缩放因子

    def init_weights(self):
        if not self.use_dcn:
            for m in self.cls_convs:
                normal_init(m.conv, std=0.01)
            for m in self.reg_convs:
                normal_init(m.conv, std=0.01)
            for m in self.coordinate_convs:
                if not isinstance(m, NonLocalModule):
                    normal_init(m.conv,
                                std=0.01)  # add for 2d-dimension predict
            for m in self.refine_convs:
                normal_init(m.conv, std=0.01)
            if self.additional_cls_branch:
                for m in self.additional_cls_convs:
                    if isinstance(m, ConvModule):
                        normal_init(m.conv, std=0.01)
                    else:
                        normal_init(m, std=0.01)
                normal_init(self.additional_cls, std=0.01)
        else:
            pass

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.polar_cls, std=0.01, bias=bias_cls)
        normal_init(self.polar_reg, std=0.01)
        normal_init(self.coordinate, std=0.01)
        normal_init(self.refine, std=0.01)
        normal_init(self.polar_centerness, std=0.01)
        # initialization for feature fusion in refine stage
        if self.fuse_center_feat:
            normal_init(self.fuse_center, std=0.01)
        if self.fuse_global_feat:
            normal_init(self.fuse_global, std=0.01)
        if self.fuse_adj_feat:
            normal_init(self.fuse_adj.conv.fc, std=0.01)
        if self.fuse_snake_feat:
            normal_init(self.fuse_snake.compress_conv, std=0.01)
            normal_init(self.fuse_snake.fuse_conv, std=0.01)
            for m in self.fuse_snake.modules():
                if isinstance(m, SnakeBlock):
                    normal_init(m.conv.fc, std=0.01)
                    constant_init(m.norm, 1, bias=0)

        if self.cascade_refine_num > 0:
            for m in self.cascade_refine_convs:
                normal_init(m.conv, std=0.01)
            for cascade_refine_layer in self.cascade_refine_layers:
                normal_init(cascade_refine_layer, std=0.01)

    def forward(self, feats):
        '''
        feats: 长度为5的数组
        对传入的多层特征，先分层计算得到若干变量，然后将各层得到的结果按照变量进行合并
        '''
        level_id = [1, 2, 3, 4, 5]
        if self.sample_last:
            concat_feats = []
            for i in range(5):
                if i > 0:
                    concat_feats.append([feats[i], feats[i - 1]])
                else:
                    concat_feats.append([feats[i],
                                         feats[i]])  # the first layer
            return multi_apply(self.forward_single, concat_feats,
                               self.scales_bbox, self.scales_mask,
                               self.scale_refine)
        else:
            if self.cascade_refine_num == 0:
                return multi_apply(self.forward_single, feats,
                                   self.scales_bbox, self.scales_mask,
                                   self.scale_refine, level_id)
            else:
                return multi_apply(self.forward_single, feats,
                                   self.scales_bbox, self.scales_mask,
                                   self.scale_refine,
                                   self.cascade_refine_scale)

    def forward_single(self,
                       x,
                       scale_bbox,
                       scale_mask,
                       scale_refine,
                       level_id,
                       cascade_refine_scale=None):
        '''
        计算三个分支，其中分类分支最后会分别预测分类分数和centerness
        bbox预测分支：预测的是到四条边距离的指数，因为预测始终应该为正
        mask预测分支：预测的是极径的指数，因为预测始终应该为正
        angle预测分支：预测的是极角之差
        '''
        assert isinstance(
            x, list
        ) == self.sample_last  # if sample_last, input the last layer's feat map
        if not self.sample_last:
            cls_feat = x
            reg_feat = x
            mask_feat = x
            mask_refine_feat = x  # feat for refine offset
            if self.additional_cls_branch:
                if self.sample_last:
                    additional_cls_feat = x[0]
                else:
                    additional_cls_feat = x

        else:
            cls_feat = x[0]
            reg_feat = x[0]
            mask_feat = x[0]
            mask_refine_feat = x[1]

        b, c, h, w = mask_feat.shape

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.polar_cls(cls_feat)  # 只是分数，并未换成概率
        centerness = self.polar_centerness(cls_feat)  # 只是数值

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)  # 逐层运算
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale_bbox(self.polar_reg(reg_feat)).float().exp()

        intermediate_feat = []
        for coordinate_conv in self.coordinate_convs:
            mask_feat = coordinate_conv(mask_feat)
            if self.fuse_intermediate_feat:
                intermediate_feat.append(mask_feat)
        res_mask = self.coordinate(
            mask_feat)  # 尺寸：(batch,72, h, w) or 极径: (batch, 36, h, w)

        # 对预测偏移进行尺度的缩放
        if (not self.polar_xy) or self.polar_xy_represent:
            '''
            第一阶段预测polar的二维坐标形式或第一阶段预测等间距轮廓点的二维偏移
            '''
            shape = res_mask.shape
            res_mask = res_mask[:, :, None, :, :]
            res_mask = res_mask.contiguous().view(b, 36, 2, h,
                                                  w).permute(0, 2, 1, 3,
                                                             4)  # (N,2,36,H,W)
            xy_pred_init = scale_mask(
                res_mask).float()  # 输出为(batch,2,36,h,w)，表示初始偏移offset1
            xy_pred_init = xy_pred_init.permute(0, 2, 1, 3,
                                                4)  # 尺寸为(batch,36,2,h,w)
        else:
            '''
            第一阶段预测的是polar的一维形式————极径
            '''
            # 先对极径进行缩放后，转为表示偏移二维坐标(batch,36,2,h,w)
            polar_pred_init = scale_mask(
                res_mask).float().exp()  # (batch, 36, h, w)
            shape = polar_pred_init.shape
            sin = torch.sin(self.angles)
            cos = torch.cos(self.angles)
            sin = sin[None, :, None, None].repeat(shape[0], 1, shape[2],
                                                  shape[3])
            cos = cos[None, :, None, None].repeat(shape[0], 1, shape[2],
                                                  shape[3])
            x = polar_pred_init * sin  # (batch,36,h,w)
            y = polar_pred_init * cos
            x = x[:, :, None, :, :]
            y = y[:, :, None, :, :]
            xy_pred_init = torch.cat([x, y], 2)  # (batch, 36, 2, h, w)

        # 通过初次预测的位置的特征来精修偏移
        # 这个偏移是基于整张图像的，但采样的faeture map是下采样后的，应该将这个预测坐标也缩放到相同大小，再加到grid网格上
        if not self.sample_last:
            downsample_rate = 1280 / xy_pred_init.shape[-1]
            # 兼容多尺度训练
            # downsample_rate = (2**(level_id + 2)) * w / xy_pred_init.shape[-1]
            # print(downsample_rate)
            if self.polar_xy:
                if not self.polar_xy_represent:
                    offset1 = xy_pred_init / downsample_rate  # 此处polar预测的极径是没有归一化的
                else:
                    offset1 = xy_pred_init / downsample_rate / self.normalize_factor  # 在使用极径的二维偏移形式时仍然是使用了归一化的
            else:
                offset1 = xy_pred_init / downsample_rate / self.normalize_factor
        else:
            # TODO：目前不适用于polar_xy
            print('sample features from the last feature map')
            downsample_rate = 1280 / xy_pred_init.shape[-1]
            expand_rate = mask_refine_feat.shape[-1] / xy_pred_init.shape[
                -1]  # 除了P3，其他均在上一层采样
            offset1 = xy_pred_init / downsample_rate / self.normalize_factor * expand_rate

        ## 减少精细回归对初始回归的梯度影响
        offset1_detach = (1 - self.gradient_mul) * offset1.detach(
        ) + self.gradient_mul * offset1  # (batch, 36, 2, h, w)

        # # 测试compute_offset_feature函数
        # a = torch.cat(
        #     [torch.zeros(1, 36, 1, 96, 160),
        #      torch.zeros(1, 36, 1, 96, 160)],
        #     axis=2)
        # a[0, 18, 0, :, :] = 1
        # a[0, 18, 1, :, :] = 2
        # a = a.cuda()
        # cc = torch.zeros_like(mask_refine_feat)
        # cc[0, :, 46, 79] = 1
        # b = self.compute_offset_feature(cc, a, padding_mode='border')
        # print(
        #     b[0, 18, :, 44, 78]
        # )  # 设当前中心为(44,78),我们预测出其中一个轮廓点的偏移为(1,2)((x,y)形式)，然后根据此偏移，我们会采样(46,79)的特征来填充(44,78)，取到的是轮廓上的点

        if not self.predict_refine_first:
            # 在对显存需要大的时候使用compute_feature
            if self.fuse_intermediate_feat:
                refine_feat = self.compute_features(
                    mask_refine_feat, offset1_detach,
                    padding_mode='border')  # (N,36,C,H,W)
            else:
                refine_feat = self.compute_offset_feature(
                    mask_refine_feat, offset1_detach,
                    padding_mode='border')  # (N,36,C,H,W)


            #融合更多的mask分支上的特征
            if self.fuse_intermediate_feat:
                states = [refine_feat]
                for i in self.intermediate_feat_level:
                    # refine_feat = self.compute_offset_feature(
                    #     intermediate_feat[i],
                    #     offset1_detach,
                    #     padding_mode='border')
                    refine_feat = self.compute_features(intermediate_feat[i],
                                                        offset1_detach,
                                                        padding_mode='border')
                    states.append(refine_feat)  # (N,36,C,H,W)

                refine_feat = self.fuse_intermediate(states)  # (N,36,C,H,W)

            # 每个特征融合模块的输出应该为(N,36,C,H,W)
            # concat中心点特征和各轮廓点特征，尺寸：(N,36,2C,H,W), 再使用一维卷积降通道至(NHW,C,36)
            if self.fuse_center_feat:
                center_feat = mask_refine_feat[:, None, :, :, :].expand_as(
                    refine_feat)  # (1, 36, 256, 96, 160)
                refine_feat = torch.cat([refine_feat, center_feat],
                                        axis=2)  # (1, 36, 512, 96, 160)
                refine_feat = refine_feat.permute(0, 3, 4, 2,
                                                  1)  # (N,H,W,2C,36)
                refine_feat = refine_feat.contiguous().view(
                    -1, 2 * self.feat_channels, 36)  # (NHW, 2*256, 36)
                refine_feat = self.fuse_center(refine_feat)  # (NHW, 256, 36)
                refine_feat = refine_feat.view(b, h, w, self.feat_channels,
                                               36).permute(0, 4, 3, 1,
                                                           2)  # (N,36,256,H,W)

            # 将各轮廓点的二维偏移添加到通道维,类似于coordconv，是detach的
            if self.fuse_coord_feat:
                if not self.coord_radius:
                    deviation_coord_feat = offset1.detach(
                    )  # (batch, 36, 2, h, w)
                    refine_feat = torch.cat(
                        [refine_feat, deviation_coord_feat],
                        axis=2)  # (batch, 36, 258, h, w)
                else:
                    deviation_coord_feat = polar_pred_init.detach(
                    )[:, :, None, :, :]  # (batch, 36, 1, h, w)
                    refine_feat = torch.cat(
                        [refine_feat, deviation_coord_feat],
                        axis=2)  # (batch, 36, 257, h, w)

            # 融合每个轮廓点周围点的信息
            if self.fuse_adj_feat:
                refine_feat = refine_feat.permute(
                    0, 3, 4,
                    1, 2).contiguous().view(b * h * w, 36, -1).permute(
                        0, 2, 1)  # (bhw, 256, 36) / (bhw, 258, 36)
                # 此处维度不一致无法相加，后面增加更多snake时可采用残差连接
                if self.fuse_coord_feat:
                    refine_feat = self.fuse_adj(refine_feat)  # (bhw, 256, 36)
                else:
                    refine_feat = refine_feat + self.fuse_adj(
                        refine_feat)  # (bhw, 256, 36) # skip connection
                refine_feat = refine_feat.view(b, h, w, self.feat_channels,
                                               36).permute(0, 4, 3, 1,
                                                           2)  # （b,36,256,h,w）
            if self.fuse_snake_feat:
                refine_feat = refine_feat.permute(0, 3, 4, 1,
                                                  2).contiguous().view(
                                                      b * h * w, 36,
                                                      -1).permute(0, 2, 1)
                refine_feat = self.fuse_snake(refine_feat)
                refine_feat = refine_feat.view(b, h, w, self.feat_channels,
                                               36).permute(0, 4, 3, 1,
                                                           2)  # （b,36,256,h,w）

            # 融合所在轮廓的全局信息
            if self.fuse_global_feat:
                global_feat = torch.max(
                    refine_feat, dim=1,
                    keepdim=True)[0]  # (b, 1, 256, h, w) / (b, 1, 258, h, w)
                global_feat = global_feat.expand_as(
                    refine_feat)  # (b, 36, 256, h, w) / (b, 36, 258, h, w)
                refine_feat = torch.cat(
                    [refine_feat, global_feat],
                    axis=2)  # (b, 36, 512, h, w)  / (b, 1, 516, h, w)
                refine_feat = refine_feat.permute(
                    0, 3, 4, 2, 1)  # (N,H,W,512,36) / (b,  h, w, 516, 36)
                refine_feat = refine_feat.contiguous().view(
                    b * h * w, -1,
                    36)  # (NHW, 2*256, 36) / (NHW, 2*258, 36) 降通道
                refine_feat = self.fuse_global(refine_feat)  # (NHW, 256, 36)
                refine_feat = refine_feat.view(b, h, w, self.feat_channels,
                                               36).permute(0, 4, 3, 1,
                                                           2)  # (N,36,256,h,w)

            refine_feat = refine_feat.contiguous().view(b, -1, h,
                                                        w)  # (N,36C,H,W)

            for refine_layer in self.refine_convs:
                refine_feat = refine_layer(refine_feat)  # 分组卷积，仅各点组内卷积

            if not self.refine_1d:
                res_mask = self.refine(
                    refine_feat)  # (N,72,H,W) or 第二阶段仍预测极径：(N,36,H,W)
            else:
                refine_feat = refine_feat.contiguous().view(b, 36, -1, h, w)
                refine_feat = refine_feat.permute(0, 2, 1, 3,
                                                  4).contiguous().view(
                                                      b,
                                                      -1,
                                                      36 * h * w,
                                                  )  # (b,c, 36hw)
                res_mask = self.refine(refine_feat)  # (b, 1, 36hw)
                res_mask = res_mask.squeeze(dim=1).contiguous().view(
                    b, 36, h, w)
            #####################################

            # 根据第二阶段的预测为二维或一维来区分
            if not self.polar_both:
                '''
                第一阶段预测极径(1d)，第二阶段预测极径的二维偏移(2d)；
                第一阶段预测极径的二维偏移(2d)，第二阶段预测等间距的二维偏移(2d)；
                第一阶段预测等间距的二维偏移(2d)，第二阶段预测等间距的二维偏移(2d)；
                第一阶段预测极径的二维偏移(2d)，第二阶段预测极径的二维偏移(2d)；
                '''
                res_mask = res_mask.contiguous().view(b, 36, 2, h, w).permute(
                    0, 2, 1, 3, 4)  # (N,2,36,H,W)
                res_mask = scale_refine(
                    res_mask).float()  # 输出结果就是归一化后的值，因此此处不需要归一化
                res_mask = res_mask.permute(0, 2, 1, 3,
                                            4)  # 尺寸为(batch,36,2,h,w)
                xy_pred_refine = xy_pred_init + res_mask
            else:
                '''
                第一阶段、第二阶段同时预测极径(一维向量)
                '''
                offset2 = scale_refine(res_mask).float()
                polar_pred_refine = polar_pred_init + offset2  # (batch,36,h,w)
                polar_pred_refine = self.relu(
                    polar_pred_refine)  # 防止经过精修后的极径变为0
                # polar_pred_refine = max(1e-9, polar_pred_refine)

        else:
            for refine_layer in self.refine_convs:
                refine_feat = refine_layer(mask_refine_feat)
            offset2 = self.refine(refine_feat)  # (N,72,H,W)
            offset2 = offset2.contiguous().view(b, 36, 2, h,
                                                w).permute(0, 2, 1, 3,
                                                           4)  # (N,2,36,H,W)
            offset2 = scale_refine(offset2).float() / self.normalize_factor
            # offset2 = scale_refine(offset2).float()
            res_mask = offset2.permute(0, 2, 1, 3, 4)  # 尺寸为(batch,36,2,h,w)
            res_mask = self.compute_offset_feature(
                res_mask.contiguous().view(-1, 2, h, w),
                offset1_detach.contiguous().view(-1, 2, h, w),
                padding_mode='border')  #(batch*36,2,h,w)
            res_mask = res_mask.view(b, -1, 2, h, w)
            xy_pred_refine = xy_pred_init + res_mask

        if self.gt_unormalize:
            xy_pred_init = xy_pred_init / self.normalize_factor
            xy_pred_refine = xy_pred_refine / self.normalize_factor

        # 对偏移进行级联
        if self.cascade_refine_num > 0:
            xy_pred_refine_list = [xy_pred_refine]
            for refine_layer in self.cascade_refine_layers:
                offset = xy_pred_refine_list[
                    -1] / self.normalize_factor / downsample_rate
                offset_detach = 0.9 * offset.detach() + 0.1 * offset
                refine_feat = self.compute_offset_feature(
                    mask_refine_feat, offset_detach,
                    padding_mode='border').view(b, -1, h, w)
                for refine_conv in self.cascade_refine_convs:
                    refine_feat = refine_conv()  # 分组卷积，仅各点组内卷积
                res_mask = refine_layer(
                    refine_feat)  # (N,72,H,W)，predict normalized value
                res_mask = res_mask.contiguous().view(b, 36, 2, h, w).permute(
                    0, 2, 1, 3, 4)  # (N,2,36,H,W)
                res_mask = cascade_refine_scale(res_mask).float()
                res_mask = res_mask.permute(0, 2, 1, 3,
                                            4)  # 尺寸为(batch,36,2,h,w)
                if self.gt_unormalize:
                    xy_pred_refine = xy_pred_refine_list[
                        -1] + res_mask / self.normalize_factor
                else:
                    xy_pred_refine = xy_pred_refine_list[-1] + res_mask
                xy_pred_refine_list.append(xy_pred_refine)

        if self.additional_cls_branch:
            # if self.sample_last:
            #     additional_cls_feat = x[0]
            # else:
            #     additional_cls_feat = x
            # print(additional_cls_feat.mean())
            if self.additional_cls_single:
                # if (h, w) == (96, 160):
                if level_id == 1:
                    # 上采样
                    if self.heatmap_upsample:
                        if not self.heatmap_upsample_deconv:
                            additional_cls_feat = F.interpolate(
                                additional_cls_feat,
                                scale_factor=2,
                                mode='bilinear',
                                align_corners=False)
                    for additional_cls_layer in self.additional_cls_convs:
                        additional_cls_feat = additional_cls_layer(
                            additional_cls_feat)
                    points_score_map = self.additional_cls(additional_cls_feat)

            else:
                NotImplementedError

        if self.additional_cls_branch:
            # if (h, w) == (96, 160):
            if level_id == 1:
                if not self.polar_xy:
                    # 以前的初始init+refine+heatmap
                    return cls_score, bbox_pred, centerness, xy_pred_init, xy_pred_refine, points_score_map
                else:
                    if self.polar_xy_represent:
                        # init: polar(2d) refine:
                        return cls_score, bbox_pred, centerness, xy_pred_init, xy_pred_refine, points_score_map
                    else:
                        # init: polar(1d) refine: polar(1d)
                        return cls_score, bbox_pred, centerness, polar_pred_init, polar_pred_refine, points_score_map

            else:
                if (self.polar_xy and not self.polar_xy_represent):
                    return cls_score, bbox_pred, centerness, polar_pred_init, polar_pred_refine, None
                else:
                    return cls_score, bbox_pred, centerness, xy_pred_init, xy_pred_refine, None
        else:
            if self.polar_xy:
                # 第一阶段极径
                if not self.polar_xy_represent:
                    if not self.polar_both:
                        # 第一阶段预测极径(1-d)，第二阶段为等间距轮廓点(2-d)
                        print('test regress 2d-deviation-format polar gt')
                        return cls_score, bbox_pred, centerness, polar_pred_init, xy_pred_refine, 0  # (batch,36, h, w) (batch,36, 2, h, w)
                    else:
                        # 第一阶段预测极径(1-d)，第二阶段预测极径(1-d)
                        return cls_score, bbox_pred, centerness, polar_pred_init, polar_pred_refine, 0
                        ###############
                else:
                    # 第一阶段为polar gt的二维偏移， 第二阶段均为等间距轮廓点的二维偏移
                    # 第一阶段为polar gt的二维偏移， 第二阶段均为polar gt的二维偏移
                    return cls_score, bbox_pred, centerness, xy_pred_init, xy_pred_refine, 0  # (batch,36, 2, h, w) (batch,36, 2, h, w)
            else:
                # 两个阶段均预测等间距轮廓点的偏移
                if self.cascade_refine_num == 0:
                    return cls_score, bbox_pred, centerness, xy_pred_init, xy_pred_refine, 0  # 并不计算score map，
                else:
                    return cls_score, bbox_pred, centerness, xy_pred_init, xy_pred_refine_list, 0

################ from dense reppoints ################

    def sample_offset(self, x, flow, padding_mode):
        """
        sample feature based on offset
            Args:
                x (Tensor): input feature, size (n, c, h, w)
                flow (Tensor): flow fields, size(n, 2, h', w')
                padding_mode (str): grid sample padding mode, 'zeros' or 'border'
            Returns:
                Tensor: warped feature map (n, c, h', w')
        """
        # assert x.size()[-2:] == flow.size()[-2:]
        n, _, h, w = flow.size()  # [36, 2, 96, 160]
        # [[0..135],[0..135],[0..135]...] stack by x
        x_ = torch.arange(w).view(1, -1).expand(h, -1)  # 最多移动159
        # [[0..99],[0..99],[0..99]...] stack by y
        y_ = torch.arange(h).view(-1, 1).expand(-1, w)  # 最多移动96
        grid = torch.stack([x_, y_], dim=0).float().cuda()
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
        grid = grid + flow
        gx = 2 * grid[:, 0, :, :] / (w - 1) - 1  # normalize
        gy = 2 * grid[:, 1, :, :] / (h - 1) - 1  # normalize
        grid = torch.stack([gx, gy], dim=1)
        grid = grid.permute(0, 2, 3, 1)

        if torch.__version__ >= '1.3.0':
            return F.grid_sample(x,
                                 grid,
                                 padding_mode=padding_mode,
                                 align_corners=True)
        else:
            return F.grid_sample(x, grid, padding_mode=padding_mode)

################ from dense reppoints ################

    def compute_offset_feature(self, x, offset, padding_mode):
        """
        sample feature based on offset

            Args:
                x (Tensor) : feature map, size (n, C, h, w) ,  x first #(bn,2,h,w) 
                offset (Tensor) : offset, size (n, sample_pts*2, h, w) or (n, sample_pts, 2, h, w), x first
                padding_mode (str): 'zeros' or 'border' or 'relection'
            Returns:
                Tensor: the warped feature generated by the offset and the input feature map, size (n, sample_pts, C, h, w)
        """
        ### modify the shape
        if len(offset.shape) == 4:
            offset_reshape = offset.view(
                offset.shape[0], -1, 2, offset.shape[2],
                offset.shape[3])  # (n, sample_pts, 2, h, w)  #(bn,1,2,h,w)

        else:
            offset_reshape = offset

        num_pts = offset_reshape.shape[1]  # 1
        offset_reshape = offset_reshape.contiguous().view(
            -1, 2, offset_reshape.shape[3],
            offset_reshape.shape[4])  # (n*sample_pts, 2, h, w)   # (bn,2,h,w)
        x_repeat = x.unsqueeze(1).repeat(
            1, num_pts, 1, 1, 1)  # (n, sample_pts, C, h, w) # (bn,1,2,h,w)
        x_repeat = x_repeat.view(
            -1, x_repeat.shape[2], x_repeat.shape[3],
            x_repeat.shape[4])  # (n*sample_pts, C, h, w)  # (bn,2,h,w)

        sampled_feat = self.sample_offset(
            x_repeat, offset_reshape,
            padding_mode)  # (n*sample_pts, C, h, w) # (bn,2,h,w)
        sampled_feat = sampled_feat.view(
            -1, num_pts, sampled_feat.shape[1], sampled_feat.shape[2],
            sampled_feat.shape[3])  # (n, sample_pts, C, h, w) # (bn,1,2,h,w)

        return sampled_feat

    def compute_features(self, x, offset, padding_mode):
        """
        sample feature based on offset

            Args:
                x (Tensor) : feature map, size (n, C, h, w) ,  x first #(bn,2,h,w)
                offset (Tensor) : offset, size (n, sample_pts*2, h, w) or (n, sample_pts, 2, h, w), x first
                padding_mode (str): 'zeros' or 'border' or 'relection'
            Returns:
                Tensor: the warped feature generated by the offset and the input feature map, size (n, sample_pts, C, h, w)
        """
        if len(offset.shape) == 4:
            offset_reshape = offset.view(
                offset.shape[0], -1, 2, offset.shape[2],
                offset.shape[3])  # (n, sample_pts, 2, h, w)  #(bn,1,2,h,w)

        else:
            offset_reshape = offset

        n, num_points, _, h, w = offset_reshape.shape
        x_ = torch.arange(w).view(1, -1).expand(h, -1)  # 最多移动159
        y_ = torch.arange(h).view(-1, 1).expand(-1, w)  # 最多移动96
        grid = torch.stack([x_, y_], dim=0).float().cuda()  # (2,h,w)
        grid = grid.unsqueeze(0).expand(n, num_points, -1, -1,
                                        -1)  #(n, num_points, 2, h, w)
        grid = grid + offset
        gx = 2 * grid[:, :, 0, :, :] / (w - 1) - 1  # normalize
        gy = 2 * grid[:, :, 1, :, :] / (h - 1) - 1  # normalize
        grid = torch.stack([gx, gy], dim=2)
        grid = grid.permute(0, 2, 1, 3, 4).contiguous().view(
            offset.shape[0], 2, -1, offset_reshape.shape[4])  # (n,2,36h,w)
        grid = grid.permute(0, 2, 3, 1)  #(n,36h,w,2)
        if torch.__version__ >= '1.3.0':
            sample_feat = F.grid_sample(x,
                                        grid,
                                        padding_mode=padding_mode,
                                        align_corners=True)
        else:
            sample_feat = F.grid_sample(
                x, grid, padding_mode=padding_mode)  #(n,c,36h,w)
        sample_feat = sample_feat.view(n, -1, 36, h, w)  # (n,c,36,h,w)
        sample_feat = sample_feat.permute(0, 2, 1, 3, 4)
        return sample_feat

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'mask_init',
                          'mask_refine', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             mask_init,
             mask_refine,
             points_score_map,
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
        mask_init: 包含所有level的像素到边界的36个点的预测距离[(batch,36,2,H1,W1),(batch,36,2,H2,W2)...(batch,36,2,H5,W5)]，polar_xy为True时，是包含所有level的像素到边界的36个点极径[(batch,36,H1,W1),(batch,36,H2,W2)...(batch,36,H5,W5)]
        mask_refine: 包含所有level的像素到边界的36个点的更精确的距离[(batch,36,2，H1,W1),(batch,36,2,H2,W2)...(batch,36,2,H5,W5)]
                     mask_refine和mask_init具有相同的回归目标， polar_both为true时，是所有level的像素到边界的36个点极径[(batch,36,H1,W1),(batch,36,H2,W2)...(batch,36,H5,W5)]
        points_score_map: [(batch, H, W, 1), None, None, None, None] # self.additional_cls_single  [0,0,0,0,0]
        gt_bboxes: 未使用，[tensor(num_bbox,4),...,tensor(num_bbox,4)],list长度为batchSize,
        gt_labels: 未使用
        gt_mask: 未使用
        extra_data：包含键：_gt_labels,_gt_bboxes,_gt_xy_targets包含所有特征像素的标签、到bbox四条边的距离、36个点的偏移，并且没有归一化的值
        '''

        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(
            mask_init) == len(mask_refine)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(
            featmap_sizes, bbox_preds[0].dtype,
            bbox_preds[0].device)  # list，长度为特征的层数，所有特征点的坐标
        num_imgs = cls_scores[0].size(0)  # batch size

        # target
        # polar_xy为True时有两个关于mask的gt
        if self.additional_cls_branch:
            if not self.polar_xy:
                labels, bbox_targets, mask_targets, sample_heatmap_targets = self.polar_target(
                    all_level_points, extra_data
                )  # sample_heatmap_targets 尺寸[(H,W)....] 0或1，list长度为batch
            else:
                labels, bbox_targets, mask_targets, polar_targets, sample_heatmap_targets = self.polar_target(
                    all_level_points, extra_data
                )  # sample_heatmap_targets 尺寸[(H,W)....] 0或1，list长度为batch
            flatten_sample_heatmap = torch.cat(
                sample_heatmap_targets,
                0).contiguous().view(-1, 1)  # list —>(batch,H,W) ——> (BHW,1)
        else:
            if not self.polar_xy:
                # gt均为二维坐标, mask_targtes为点的偏移[...36,2]
                labels, bbox_targets, mask_targets = self.polar_target(
                    all_level_points, extra_data)
            else:
                labels, bbox_targets, mask_targets, polar_targets = self.polar_target(
                    all_level_points, extra_data
                )  # mask_targtes为等轮廓距离取点，polar_targets为等角度间隔取点，mask_targets对polar_both无用

        # predict
        # flatten cls_scores, bbox_preds and centerness,最终得到长度为batch的list[(num_pixel,80)...(num_pixel,80)]
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3,
                              1).contiguous().view(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]  # 放入了batch size的维度，[(num_pixel,80),(num_pixel,80)]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).contiguous().view(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).contiguous().view(-1)
            for centerness in centernesses
        ]
        if (not self.polar_xy) or self.polar_xy_represent:
            flatten_mask_init = [
                init.permute(0, 3, 4, 1, 2).contiguous().view(-1, 36, 2)
                for init in mask_init
            ]  # 36,2
        else:
            flatten_mask_init = [
                init.permute(0, 2, 3, 1).contiguous().view(-1, 36)
                for init in mask_init
            ]  # 36

        if self.cascade_refine_num == 0:
            if not self.polar_both:
                flatten_mask_refine = [
                    refine.permute(0, 3, 4, 1, 2).contiguous().view(-1, 36, 2)
                    for refine in mask_refine
                ]  # 36,2
            else:
                flatten_mask_refine = [
                    refine.permute(0, 2, 3, 1).contiguous().view(-1, 36)
                    for refine in mask_refine
                ]  # 36
        else:
            flatten_mask_refine = [[]
                                   for i in range(self.cascade_refine_num + 1)]
            for refine in mask_refine:  # length is 5
                for i in range(self.cascade_refine_num + 1):  # length is 2
                    flatten_mask_refine[i].append(refine[i].permute(
                        0, 3, 4, 1, 2).contiguous().view(-1, 36, 2))

        flatten_cls_scores = torch.cat(flatten_cls_scores)  # [num_pixel, 80]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)  # [num_pixel, 4]
        flatten_mask_init = torch.cat(
            flatten_mask_init)  # [num_pixel, 36,2] // [num_pixel, 36]
        if self.cascade_refine_num == 0:
            flatten_mask_refine = torch.cat(
                flatten_mask_refine
            )  # [num_pixel, 36,2] or polar_both=True [num_pixel, 36]
        else:

            flatten_mask_refine = [
                torch.cat(flatten_mask_refine_single)
                for flatten_mask_refine_single in flatten_mask_refine
            ]  # [[20460, 36, 2],[20460, 36, 2]]

        flatten_centerness = torch.cat(flatten_centerness)  # [num_pixel]

        # target
        flatten_labels = torch.cat(labels).long()  # [num_pixel]
        flatten_bbox_targets = torch.cat(bbox_targets)  # [num_pixel, 4]
        if self.polar_xy:
            flatten_polar_targets = torch.cat(
                polar_targets)  # [num_pixel, 36] or [num_pixel, 36, 2]
        flatten_mask_targets = torch.cat(mask_targets)  # [num_pixel, 36，2]
        flatten_points = torch.cat([
            points.repeat(num_imgs, 1) for points in all_level_points
        ])  # [batch*num_pixel,2] 放入batch size的维度

        pos_inds = flatten_labels.nonzero().contiguous().view(-1)
        num_pos = len(pos_inds)

        loss_dict = {}
        loss_cls = self.loss_cls(flatten_cls_scores,
                                 flatten_labels,
                                 avg_factor=num_pos +
                                 num_imgs)  # avoid num_pos is 0
        loss_dict['loss_cls'] = loss_cls
        if self.additional_cls_branch and self.additional_cls_single:
            points_score_map = points_score_map[0].contiguous().view(-1, 1)
            loss_dict['loss_heatmap'] = self.loss_heatmap(
                points_score_map,
                flatten_sample_heatmap,
                avg_factor=flatten_sample_heatmap.sum() +
                num_imgs)  # heatmap拉伸为列进行计算
        else:
            NotImplementedError

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]  # 预测的centerness
        pos_mask_init = flatten_mask_init[pos_inds]  # 预测的粗略回归
        if self.cascade_refine_num == 0:
            pos_mask_refine = flatten_mask_refine[pos_inds]  # 预测的精细回归
        else:
            pos_mask_refine = [
                flatten_mask_refine_single[pos_inds]
                for flatten_mask_refine_single in flatten_mask_refine
            ]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]  # 相对中心点的偏移
            if self.polar_xy:
                pos_polar_targets = flatten_polar_targets[pos_inds]

            pos_mask_targets = flatten_mask_targets[pos_inds]
            pos_points = flatten_points[pos_inds]

            if self.centerness_base == 'boundary':
                pos_centerness_targets = self.polar_centerness_target(
                    pos_mask_targets)  # 等距离间隔取点计算出的gt
            elif self.centerness_base == 'polar':
                # 在两个阶段均预测极径时
                pos_centerness_targets = self.polar_centerness_target(
                    pos_polar_targets
                )  # (N, 36) 等角度间隔取点计算的gt 若为(N, 36, 2),则计算出极径后再算centerness

            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)

            # centerness weighted iou loss,训练阶段用centerness真值加权，验证阶段用预测的centerness
            loss_bbox = self.loss_bbox(pos_decoded_bbox_preds,
                                       pos_decoded_target_preds,
                                       weight=pos_centerness_targets,
                                       avg_factor=pos_centerness_targets.sum())
            loss_dict['loss_bbox'] = loss_bbox

            # 二维损失
            # pos_mask_preds = pos_mask_preds.reshape(
            #     -1, 72)  # 将[num_pixel, 36，2]转为[num_pixel, 72]
            # pos_mask_targets = pos_mask_targets.reshape(
            #     -1, 72)  # 将[num_pixel, 36，2]转为[num_pixel, 72]
            # print('preds', pos_mask_preds)
            # print('targets', pos_mask_targets)
            if not self.polar_xy:
                loss_mask_init = self.loss_mask_init(
                    pos_mask_init,
                    pos_mask_targets,
                    weight=pos_centerness_targets,
                    avg_factor=pos_centerness_targets.sum())

            else:
                ##如果预测的polar是用二维表示的，数据集中传入gt也是二维表示的
                loss_mask_init = self.loss_mask_init(
                    pos_mask_init,
                    pos_polar_targets,
                    weight=pos_centerness_targets,
                    avg_factor=pos_centerness_targets.sum())  # (N, 36)
            loss_dict['loss_mask_init'] = loss_mask_init

            a = pos_mask_targets / self.normalize_factor

            if self.cascade_refine_num == 0:
                if not self.polar_both:
                    if not self.polarxy_polarxy:
                        loss_mask_refine = self.loss_mask_refine(
                            pos_mask_refine,
                            pos_mask_targets,
                            weight=pos_centerness_targets,
                            avg_factor=pos_centerness_targets.sum())
                    else:
                        # init: polar(2d) refine: polar(2d)
                        loss_mask_refine = self.loss_mask_refine(
                            pos_mask_refine,
                            pos_polar_targets,
                            weight=pos_centerness_targets,
                            avg_factor=pos_centerness_targets.sum())
                else:
                    loss_mask_refine = self.loss_mask_refine(
                        pos_mask_refine,
                        pos_polar_targets,
                        weight=pos_centerness_targets,
                        avg_factor=pos_centerness_targets.sum())
                loss_dict['loss_mask_refine'] = loss_mask_refine

            else:
                # 原有1个精修，级联是在此基础上增加的
                for i in range(self.cascade_refine_num + 1):
                    loss_mask_refine = self.loss_refine_weight[
                        i] * self.loss_mask_refine(
                            pos_mask_refine[i],
                            pos_mask_targets,
                            weight=pos_centerness_targets,
                            avg_factor=pos_centerness_targets.sum())
                    loss_dict['loss_mask_refine_{}'.format(
                        i)] = loss_mask_refine

            # 将预测的初始点、精修点转换为bbox,然后计算与左上右下两个点的距离（使用的坐标是相对当前中心建系，即为相对偏移）
            if self.loss_pts_bbox_init is not None:
                pos_bbox_init = pts2bbox(
                    pos_mask_init, pos_points)  # bbox左上右下点的相对坐标/偏移 (x1,y1,x2,y2)
                pos_decoded_bbox_preds2 = distance2bbox(pos_points, pos_bbox_preds)
                pos_decoded_target_preds2 = distance2bbox(pos_points,
                                                     pos_bbox_targets)
                loss_pts_bbox_init = self.loss_bbox(pos_decoded_bbox_preds2,
                                       pos_decoded_target_preds2,
                                       weight=pos_centerness_targets,
                                       avg_factor=pos_centerness_targets.sum())
                loss_dict['loss_pts_bbox_init'] = loss_pts_bbox_init
                # # pos_bbox_targets为四个相对偏移，均取正值
                # loss_pts_bbox_init = self.loss_pts_bbox_init(
                #     pos_bbox_init,
                #     pos_bbox_targets,
                #     normalize_factor=self.normalize_factor,
                #     weight=pos_centerness_targets,
                #     avg_factor=pos_centerness_targets.sum())  # (xi,y1,x2,y2)
                # loss_dict['loss_pts_bbox_init'] = loss_pts_bbox_init
            if self.loss_pts_bbox_refine is not None:
                pos_bbox_refine = pts2bbox(
                    pos_mask_refine, pos_points)  # bbox左上右下点的相对坐标/偏移 (x1,y1,x2,y2) (N,4)
                loss_pts_bbox_refine = self.loss_pts_bbox_refine(
                    pos_bbox_refine,
                    pos_bbox_targets,
                    normalize_factor=self.normalize_factor,
                    weight=pos_centerness_targets,
                    avg_factor=pos_centerness_targets.sum())  # (xi,y1,x2,y2)
                loss_dict['loss_pts_bbox_refine'] = loss_pts_bbox_refine

            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
            loss_dict['loss_centerness'] = loss_centerness

        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_mask_init = pos_mask_init.sum()
            if self.cascade_refine_num == 0:
                loss_mask_refine = pos_mask_refine.sum()
            else:
                loss_mask_refine = [
                    pos_mask_refine_single.sum()
                    for pos_mask_refine_single in pos_mask_refine
                ]
            loss_centerness = pos_centerness.sum()
            loss_dict['loss_bbox'] = loss_bbox
            loss_dict['loss_mask_init'] = loss_mask_init
            if self.cascade_refine_num == 0:
                loss_dict['loss_mask_refine'] = loss_mask_refine
            else:
                for i in range(self.cascade_refine_num + 1):
                    loss_dict['loss_mask_refine_{}'.format(
                        i)] = loss_mask_refine[i]

            loss_dict['loss_centerness'] = loss_centerness
            if self.loss_pts_bbox_init is not None:
                loss_pts_bbox_init = loss_pts_bbox_init.sum()
                loss_dict['loss_pts_bbox_init'] = loss_pts_bbox_init
            if self.loss_pts_bbox_refine is not None:
                loss_pts_bbox_refine = loss_pts_bbox_refine.sum()
                loss_dict['loss_pts_bbox_refine'] = loss_pts_bbox_refine

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
            (x.contiguous().view(-1), y.contiguous().view(-1)),
            dim=-1) + stride // 2  # 坐标
        return points

    def polar_target(self, points, extra_data):
        '''
        extra_data:{_gt_labels:,_gt_bboxes:,_gt_xy_targets:} 或
        {_gt_labels:,_gt_bboxes:,_gt_xy_targets:，_gt_polar_targets:}或
        {_gt_labels:,_gt_bboxes:,_gt_xy_targets:,_gt_sample_heatmap}
        返回坐标
        '''
        assert len(points) == len(self.regress_ranges)

        num_levels = len(points)

        if self.additional_cls_branch:
            if not self.polar_xy:
                labels_list, bbox_targets_list, mask_targets_list, sample_heatmap_lvl_list = extra_data.values(
                )  # 此处mask_targets即为xy坐标的相对偏移
            else:
                labels_list, bbox_targets_list, mask_targets_list, polar_targets_list, sample_heatmap_lvl_list = extra_data.values(
                )

            # # 可视化实例轮廓制作的gt
            # cv2.imwrite(
            #     'demo.jpg', sample_heatmap_lvl_list[0].sum(0).clamp(
            #         0, 1).cpu().numpy().astype(np.int64) * 255)
            # import pdb
            # pdb.set_trace()

            if self.all_instances_heatmap:
                sample_heatmap_lvl_list = [
                    sample_heatmap_lvl.sum(0).clamp(0, 1)[None, :, :]
                    for sample_heatmap_lvl in sample_heatmap_lvl_list
                ]  #  # 求每张sample_heatmap的的各层之和
            else:
                # 仅使用P3层的实例
                sample_heatmap_lvl_list = [
                    sample_heatmap_lvl[0, :, :][None, :, :]
                    for sample_heatmap_lvl in sample_heatmap_lvl_list
                ]  # [(96,160),...,(96,160)] list长度为batch # 取出每张sample_heatmap的的第0个

        else:
            if not self.polar_xy:
                labels_list, bbox_targets_list, mask_targets_list = extra_data.values(
                )  # 此处mask_targets即为xy坐标的相对偏移
            else:
                labels_list, bbox_targets_list, mask_targets_list, polar_targets_list = extra_data.values(
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
        ]  # 尺寸:2,5,像素个数,36,2
        if self.polar_xy:
            polar_targets_list = [
                polar_targets.split(num_points, 0)
                for polar_targets in polar_targets_list
            ]  # 尺寸:2,5,像素个数,36 / 当self.polar_xy为True且self.polar_xy_represent为True时，尺寸:2,5,像素个数,36,2

        # concat per level image,将不同图相同level的gt concat
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_mask_targets = []
        concat_lvl_polar_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
            concat_lvl_mask_targets.append(
                torch.cat(
                    [mask_targets[i] for mask_targets in mask_targets_list]))
            if self.polar_xy:
                concat_lvl_polar_targets.append(
                    torch.cat([
                        polar_targets[i]
                        for polar_targets in polar_targets_list
                    ]))

        if self.additional_cls_branch:
            if not self.polar_xy:
                return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_mask_targets, sample_heatmap_lvl_list
            else:
                return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_mask_targets, concat_lvl_polar_targets, sample_heatmap_lvl_list
        else:
            if not self.polar_xy:
                return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_mask_targets
            else:
                return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_mask_targets, concat_lvl_polar_targets  # 等间隔取点、等角度间隔取点

    def polar_centerness_target(self, pos_mask_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        if self.centerness_base == 'boundary':
            # 在预测二维坐标时，需要根据gt计算得到极径，得到当前点的centerness
            if not self.gt_unormalize:  # 当gt未归一化时，传入的pos_mask_targets也是没有归一化的结果
                if self.normalize_factor < 1:
                    pos_mask_targets = pos_mask_targets / self.normalize_factor

            dist_centers_contour = torch.sqrt((pos_mask_targets**2).sum(2))
            centerness_targets = (dist_centers_contour.min(dim=-1)[0] /
                                  dist_centers_contour.max(dim=-1)[0])
            return torch.sqrt(centerness_targets)

        elif self.centerness_base == 'polar':
            if len(pos_mask_targets.shape) == 2:
                '''polar轮廓点用极径表示'''
                centerness_targets = (pos_mask_targets.min(dim=-1)[0] /
                                      pos_mask_targets.max(dim=-1)[0])
            elif len(pos_mask_targets.shape) == 3:
                '''polar轮廓点用二维偏移表示'''
                if not self.gt_unormalize:  # 当gt未归一化时，传入的pos_mask_targets也是没有归一化的结果
                    if self.normalize_factor < 1:
                        pos_mask_targets = pos_mask_targets / self.normalize_factor
                dist_centers_contour = torch.sqrt((pos_mask_targets**2).sum(2))
                centerness_targets = (dist_centers_contour.min(dim=-1)[0] /
                                      dist_centers_contour.max(dim=-1)[0])

            return torch.sqrt(centerness_targets)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   xy_pred_init,
                   xy_pred_refine,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        # 用精修偏移中的最后一层作为精修阶段的预测
        if self.cascade_refine_num > 0:
            xy_pred_refine = [
                xy_pred_refine_single[-1]
                for xy_pred_refine_single in xy_pred_refine
            ]

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
            xy_pred_init_list = [
                xy_pred_init[i][img_id].detach() for i in range(num_levels)
            ]
            xy_pred_refine_list = [
                xy_pred_refine[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(
                cls_score_list, bbox_pred_list, xy_pred_init_list,
                xy_pred_refine_list, centerness_pred_list, mlvl_points,
                img_shape, scale_factor, cfg, rescale
            )  # tuple(det_bboxes, det_labels, det_masks) # (center,radius,angles)
            result_list.append(
                det_bboxes)  # [(det_bboxes, det_labels, det_masks),()...]
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          xy_preds_init,
                          xy_preds_refine,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        '''
        :cls_scores: [tensor(80,H1,W1),tensor,...tensor]长度为5的数组，每个tensor表示对应尺寸的各个点对应的预测结果
        :bbox_preds: [tensor(4,H1,W1),tensor,...tensor]长度为5的数组，每个tensor表示对应尺寸的各个点对应的预测结果
        :xy_preds_init: [tensor(36,2,H1,W1),tensor,...tensor]长度为5的数组，每个tensor表示对应尺寸的各个点对应的预测粗略偏移 // [tensor(36,H1,W1),tensor,...tensor]长度为5的数组，每个tensor表示对应尺寸的各个点对应的极径
        :xy_preds_refine: [tensor(36,2,H1,W1),tensor,...tensor]长度为5的数组，每个tensor表示对应尺寸的各个点对应的的预测总偏移(精修偏移)
        需要输出的det_masks为(保留的bbox数目,2,36)
        '''
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_masks_init = []
        mlvl_masks_refine = []
        mlvl_centerness = []
        # add by amd
        mlvl_center = []

        # 遍历每个level的预测结果
        for cls_score, bbox_pred, xy_pred_init, xy_pred_refine, centerness, points in zip(
                cls_scores, bbox_preds, xy_preds_init, xy_preds_refine,
                centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).contiguous().view(
                -1, self.cls_out_channels).sigmoid()

            centerness = centerness.permute(1, 2,
                                            0).contiguous().view(-1).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).contiguous().view(-1, 4)
            if not self.polar_xy:
                xy_pred_init = xy_pred_init.permute(
                    2, 3, 0,
                    1).contiguous().view(-1, 36, 2).permute(0, 2,
                                                            1)  # 输出需要尺寸为（2，36）
            else:
                if not self.polar_xy_represent:
                    xy_pred_init = xy_pred_init.permute(
                        1, 2, 0).contiguous().view(-1, 36)  # (num_pixels, 36)
                else:
                    # 二维偏移极径
                    xy_pred_init = xy_pred_init.permute(
                        2, 3, 0,
                        1).contiguous().view(-1, 36,
                                             2).permute(0, 2,
                                                        1)  # 输出需要尺寸为（2，36）
            if not self.polar_both:
                xy_pred_refine = xy_pred_refine.permute(
                    2, 3, 0,
                    1).contiguous().view(-1, 36, 2).permute(0, 2,
                                                            1)  # 输出需要尺寸为（2，36）
            else:
                xy_pred_refine = xy_pred_refine.permute(
                    1, 2, 0).contiguous().view(-1, 36)  # (num_pixels, 36)

            nms_pre = cfg.get('nms_pre', -1)
            # 在每张图上的实例数目大于1000时，进行非极大值抑制
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(
                    dim=1)  # 选出每个实例分类分数最高的类
                _, topk_inds = max_scores.topk(nms_pre)  # 选出每张图上分类分数的前1000
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                xy_pred_init = xy_pred_init[topk_inds, :]
                xy_pred_refine = xy_pred_refine[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)

            # 将预测的偏移加回中心点
            if self.polar_xy:
                # 第一阶段预测的polar不是(x,y)的形式
                if not self.polar_xy_represent:
                    sin = torch.sin(self.angles)[None, :]  # (1,36)
                    cos = torch.cos(self.angles)[None, :]
                    x = xy_pred_init * sin  # (num_pixels, 36)
                    y = xy_pred_init * cos
                    x = x[:, None, :]
                    y = y[:, None, :]
                    xy_pred_init = torch.cat([x, y], 1)  # (num_pixels, 2, 36)
                if self.polar_both:
                    sin = torch.sin(self.angles)[None, :]  # (1,36)
                    cos = torch.cos(self.angles)[None, :]
                    x = xy_pred_refine * sin  # (num_pixels, 36)
                    y = xy_pred_refine * cos
                    x = x[:, None, :]
                    y = y[:, None, :]
                    xy_pred_refine = torch.cat([x, y],
                                               1)  # (num_pixels, 2, 36)

            masks_init = distance2mask(points,
                                       xy_pred_init,
                                       max_shape=img_shape,
                                       normalize_factor=self.normalize_factor,
                                       gt_unormalize=self.gt_unormalize)
            masks_refine = distance2mask(
                points,
                xy_pred_refine,
                max_shape=img_shape,
                normalize_factor=self.normalize_factor,
                gt_unormalize=self.gt_unormalize)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_masks_init.append(masks_init)
            mlvl_masks_refine.append(masks_refine)
            # add by amd
            mlvl_center.append(points)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_masks_init = torch.cat(mlvl_masks_init)
        mlvl_masks_refine = torch.cat(mlvl_masks_refine)
        # add by amd
        mlvl_center = torch.cat(mlvl_center)  #(3260,2)

        if rescale:
            _mlvl_bboxes = mlvl_bboxes / mlvl_bboxes.new_tensor(scale_factor)
            try:
                scale_factor = torch.Tensor(scale_factor)[:2].cuda().unsqueeze(
                    1).repeat(1, 36)
                _mlvl_masks_init = mlvl_masks_init / scale_factor
                _mlvl_masks_refine = mlvl_masks_refine / scale_factor
                ###
                scale_factor_center = scale_factor[:, 0].unsqueeze(0)
                _mlvl_center = mlvl_center / scale_factor_center # scale is not accurate
                ###
            except:
                _mlvl_masks_init = mlvl_masks_init / mlvl_masks_init.new_tensor(
                    scale_factor)
                _mlvl_masks_refine = mlvl_masks_refine / mlvl_masks_refine.new_tensor(
                    scale_factor)

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
            ], -1)  # 按照mask得到bbox
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
            vis_info = {'center': _mlvl_center}
            if self.refine_mask:
                det_bboxes, det_labels, det_masks, det_centers = multiclass_nms_with_mask(
                    _mlvl_bboxes,
                    mlvl_scores,
                    _mlvl_masks_refine,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness + centerness_factor,
                    vis_info=vis_info)  # 使用原始的bbox
            else:
                det_bboxes, det_labels, det_masks, det_centers = multiclass_nms_with_mask(
                    _mlvl_bboxes,
                    mlvl_scores,
                    _mlvl_masks_init,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness + centerness_factor,
                    vis_info=vis_info)
        return det_bboxes, det_labels, det_masks, det_centers


# test
# not calculate accumlative angle, only for polarmask_double_gt_head
def distance2mask(points,
                  distances,
                  max_shape=None,
                  normalize_factor=1,
                  gt_unormalize=False):
    '''Decode distance prediction to 36 mask points
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point 分别为yx偏移（与后面的绘图一致）.尺寸为(N,2,36)
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded masks.
    '''
    num_points = points.shape[0]
    points = points[:, :, None].repeat(1, 1, 36)
    c_x, c_y = points[:, 0], points[:, 1]

    if not gt_unormalize:
        if normalize_factor < 1:
            x = distances[:, 0, :] / normalize_factor + c_x
            y = distances[:, 1, :] / normalize_factor + c_y
        else:
            x = distances[:, 0, :] + c_x
            y = distances[:, 1, :] + c_y
    else:
        x = distances[:, 0, :] + c_x
        y = distances[:, 1, :] + c_y

    if max_shape is not None:
        x = x.clamp(min=0, max=max_shape[1] - 1)
        y = y.clamp(min=0, max=max_shape[0] - 1)

    res = torch.cat([x[:, None, :], y[:, None, :]], dim=1)
    return res


def pts2bbox(pts, pos_points=None):
    '''找到36个点中的极点的延长线的交点，得到bbox左上右下的点
    Args:
        pts (Tensor): Shape (n, 36, 2) or (n, 36) 表示点相较中心的偏移，有正有负
        pos_points: (n,2)
    
    Returns:
        Tensor: bbox transformed from pts (x1, y1, x2, y2)
    '''
    if len(pts.shape) == 2:
        angles = torch.range(0, 350, 10).cuda() / 180 * math.pi
        sin = torch.sin(angles)[None, :] #(1, 36)
        cos = torch.cos(angles)[None, :]
        x = pts * sin
        y = pts * cos
        pts = torch.cat([x[:,None,:], y[:,None,:]], dim=1) 
        pts = distance2mask(pos_points, pts) # (N, 2, 36)


    x1 = pts[:, 0, :].min(dim=1)[0][:, None]  # (n,1)
    y1 = pts[:, 1, :].min(dim=1)[0][:, None]
    x2 = pts[:, 0, :].max(dim=1)[0][:, None]
    y2 = pts[:, 1, :].max(dim=1)[0][:, None]
    
    if pos_points is not None:
        delta_x1 = pos_points[:, 0, None] - x1
        delta_y1 = pos_points[:, 1, None] - y1
        delta_x2 = pos_points[:, 0, None] + x1
        delta_y2 = pos_points[:, 1, None] + y1

    pst_left_top_right_down = torch.cat([delta_x1, delta_y1, delta_x2, delta_y2], dim=1)

    return pst_left_top_right_down

