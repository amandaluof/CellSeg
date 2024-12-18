from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn as nn

import mmcv
from mmdet.core import bbox2result, bbox_mask2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from IPython import embed
import time
import torch
import cv2
import numpy as np
import pycocotools.mask as mask_util
import math
import os
from .utils import Sobel
from ..utils import Scale_channel
import torch.nn.functional as F


def vis_bbox_mask2result_xy(bboxes, masks, labels, centers, num_classes,
                            img_meta):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        masks (Tensor): shape (n, 2, 36)
        labels (Tensor): shape (n, )
        centers (Tensor): shape (n, 2)
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    ori_shape = img_meta['ori_shape']
    img_h, img_w, _ = ori_shape

    mask_results = [[] for _ in range(num_classes - 1)]

    # iterate every bbox
    for i in range(masks.shape[0]):
        im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask = [
            masks[i].transpose(1, 0).unsqueeze(1).int().data.cpu().numpy()
        ]  # convert to int
        im_mask = cv2.drawContours(im_mask, mask, -1, 1, -1)
        rle = mask_util.encode(np.array(im_mask[:, :, np.newaxis],
                                        order='F'))[0]

        label = labels[i]

        mask_results[label].append(rle)

    if bboxes.shape[0] == 0:
        bbox_results = [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
        xy_results = [
            np.zeros((0, 2, 36), dtype=np.float32)
            for i in range(num_classes - 1)
        ]
        center_results = [
            np.zeros((0, 2), dtype=np.float32) for i in range(num_classes - 1)
        ]
        return bbox_results, mask_results, xy_results, center_results

    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        xy_results = masks.cpu().numpy()
        centers = centers.cpu().numpy()
        bbox_results = [bboxes[labels == i, :] for i in range(num_classes - 1)]
        xy_results = [
            xy_results[labels == i, :] for i in range(num_classes - 1)
        ]
        center_results = [
            centers[labels == i, :] for i in range(num_classes - 1)
        ]
        return bbox_results, mask_results, xy_results, center_results


def vis_bbox_mask2result_polar(bboxes, masks, labels, centers, radius, angles,
                               num_classes, img_meta):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        masks (Tensor): shape (n, 2, 36)
        labels (Tensor): shape (n, )
        centers (Tensor): shape (n, 2)
        radius (Tensor): shape (n, 36)
        angles (Tensor): shape (n, 36)
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    ori_shape = img_meta['ori_shape']
    img_h, img_w, _ = ori_shape

    mask_results = [[] for _ in range(num_classes - 1)]

    # iterate every bbox
    for i in range(masks.shape[0]):
        im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask = [
            masks[i].transpose(1, 0).unsqueeze(1).int().data.cpu().numpy()
        ]  # convert to int
        im_mask = cv2.drawContours(im_mask, mask, -1, 1, -1)
        rle = mask_util.encode(np.array(im_mask[:, :, np.newaxis],
                                        order='F'))[0]

        label = labels[i]

        mask_results[label].append(rle)

    if bboxes.shape[0] == 0:
        bbox_results = [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
        center_results = [
            np.zeros((0, 2), dtype=np.float32) for i in range(num_classes - 1)
        ]
        radius_results = [
            np.zeros((0, 36), dtype=np.float32) for i in range(num_classes - 1)
        ]
        angle_results = [
            np.zeros((0, 36), dtype=np.float32) for i in range(num_classes - 1)
        ]
        return bbox_results, mask_results, center_results, radius_results, angle_results
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        centers = centers.cpu().numpy()
        radius = radius.cpu().numpy()
        angles = angles.cpu().numpy()
        bbox_results = [bboxes[labels == i, :] for i in range(num_classes - 1)]
        center_results = [
            centers[labels == i, :] for i in range(num_classes - 1)
        ]
        radius_results = [
            radius[labels == i, :] for i in range(num_classes - 1)
        ]
        angle_results = [
            angles[labels == i, :] for i in range(num_classes - 1)
        ]
        return bbox_results, mask_results, center_results, radius_results, angle_results


@DETECTORS.register_module
class PolarMask_Angle(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 gradient_input=False,
                 fusion=False,
                 fusion_pos=None,
                 fusion_style=None,
                 share_weight=True):
        super(PolarMask_Angle, self).__init__(backbone, neck, bbox_head,
                                              train_cfg, test_cfg, pretrained)
        self.gradient_input = gradient_input
        # self.sobel_kernel = torch.from_numpy(
        #     np.array([[-1, -1, -1], [-1, 2, -1], [-1, -1, -1]],
        #              dtype='float32').reshape((1, 1, 3, 3))).cuda()
        self.fusion = fusion  # 是否融合梯度图特征
        self.fusion_pos = fusion_pos
        self.fusion_style = fusion_style
        self.share_weight = share_weight
        if self.gradient_input:
            self.sobel_layer = Sobel()

        #逐通道加权求和
        if fusion_style == 'Add':
            self.scale1 = Scale_channel(1, 256)
            self.scale2 = Scale_channel(1, 256)

    def extract_feat(self, img):
        '''
        img: (1,3,768,1280) (channel_order='bgr')
        Gray = R/3 + G/3 + B/3
        '''
        if self.gradient_input:
            gradient_img = self.sobel_layer(img)
            gradient_img = torch.cat(
                [gradient_img, gradient_img, gradient_img],
                axis=1)  # (N,3,H,W)
        # # visualization graident img
        # import cv2
        # cv2.imwrite('img_original.jpg', img[0].permute(1, 2, 0).cpu().numpy())
        # # gray = img[:,
        # #            0, :, :] * 0.114 + img[:,
        # #                                   1, :, :] * 0.587 + img[:,
        # #                                                          2, :, :] * 0.299  # (N,h,w)
        # # cv2.imwrite('img_gray.jpg',
        # #             gray[0, :, :].cpu().numpy().astype(np.uint8))
        # # gray = gray[0, :, :].cpu().numpy().astype(np.uint8)

        # # edge = F.conv2d(gray[None, :, :],
        # #                 self.sobel_kernel,
        # #                 stride=1,
        # #                 padding=1)  # (N,1,H,W)
        # cv2.imwrite('img_edge.jpg',
        #             gradient_img[0, :, :, :].cpu().numpy().astype(np.uint8))

        # backbone 输出：[torch.Size([1, 256, 192, 320]), torch.Size([1, 512, 96, 160]), torch.Size([1, 1024, 48, 80]), torch.Size([1, 2048, 24, 40])]
        # FPN 输出: [torch.Size([1, 256, 96, 160]),torch.Size([1, 256, 48, 80]),torch.Size([1, 256, 24, 40]),torch.Size([1, 256, 12, 20]),torch.Size([1, 256, 6, 10])]

        if self.fusion:
            if self.share_weight:
                x = self.backbone(img)  # (N,C,H,W)
                gradient_x = self.backbone(gradient_img)

                fusion_outs = []
                if self.fusion_pos == 'after_backbone':
                    if self.fusion_style == 'Add':
                        for _x1, _x2 in zip(x, gradient_x):
                            fusion_outs.append(
                                self.scale1(_x1) + self.scale2(_x2))
                    x = tuple(fusion_outs)
                    if self.with_neck:
                        x = self.neck(x)
                if self.fusion_pos == 'after_fpn':
                    if self.with_neck:
                        x = self.neck(x)
                        gradient_x = self.neck(gradient_x)
                    if self.fusion_style == 'Add':
                        for _x1, _x2 in zip(x, gradient_x):
                            fusion_outs.append(
                                self.scale1(_x1) + self.scale2(_x2))
                    x = tuple(fusion_outs)
        else:
            x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      _gt_labels=None,
                      _gt_bboxes=None,
                      _gt_masks=None,
                      _gt_contour_all_points=None,
                      _gt_radius=None,
                      _gt_angles=None,
                      _gt_xy_targets=None,
                      _gt_polar_targets=None,
                      _gt_sample_heatmap=None):
        if _gt_labels is not None:
            if _gt_contour_all_points is not None:
                extra_data = dict(
                    _gt_labels=_gt_labels,
                    _gt_bboxes=_gt_bboxes,
                    _gt_masks=_gt_masks,
                    _gt_contour_all_points=_gt_contour_all_points)
            elif _gt_angles is not None:
                extra_data = dict(_gt_labels=_gt_labels,
                                  _gt_bboxes=_gt_bboxes,
                                  _gt_radius=_gt_radius,
                                  _gt_angles=_gt_angles)
            elif _gt_xy_targets is not None:
                if _gt_sample_heatmap is None:
                    if _gt_polar_targets is None:
                        extra_data = dict(_gt_labels=_gt_labels,
                                          _gt_bboxes=_gt_bboxes,
                                          _gt_xy_targets=_gt_xy_targets)
                    else:
                        extra_data = dict(_gt_labels=_gt_labels,
                                          _gt_bboxes=_gt_bboxes,
                                          _gt_xy_targets=_gt_xy_targets,
                                          _gt_polar_targets=_gt_polar_targets)

                else:
                    if _gt_polar_targets is None:
                        extra_data = dict(
                            _gt_labels=_gt_labels,
                            _gt_bboxes=_gt_bboxes,
                            _gt_xy_targets=_gt_xy_targets,
                            _gt_sample_heatmap=_gt_sample_heatmap)
                    else:
                        extra_data = dict(
                            _gt_labels=_gt_labels,
                            _gt_bboxes=_gt_bboxes,
                            _gt_xy_targets=_gt_xy_targets,
                            _gt_polar_targets=_gt_polar_targets,
                            _gt_sample_heatmap=_gt_sample_heatmap)

        else:
            extra_data = None

        x = self.extract_feat(
            img
        )  # (2,256,96,160) (2,256,48,80) (2,256,24,40) (2,256,12,20) (2,256,6,10)
        outs = self.bbox_head(x)
        # print(self.bbox_head.cls_convs[0].conv.weight[0, 0, :, :])
        # print(self.bbox_head.refine.weight[0, :10, :, :].reshape(1, -1))
        # print([a for a in self.bbox_head.scale_refine.parameters()])
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        
        losses = self.bbox_head.loss(*loss_inputs,
                                     gt_masks=gt_masks,
                                     gt_bboxes_ignore=gt_bboxes_ignore,
                                     extra_data=extra_data)

        return losses

    def simple_test(self, img, img_meta, rescale=False, show=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        outs = outs[:-1]
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(
            *bbox_inputs
        )  # det_bboxes, det_labels, det_masks, det_centers, det_radius, det_angles

        if len(bbox_list[0]) == 6:
            results = [
                vis_bbox_mask2result_polar(det_bboxes, det_masks, det_labels,
                                           det_centers, det_radius, det_angles,
                                           self.bbox_head.num_classes,
                                           img_meta[0])
                for det_bboxes, det_labels, det_masks, det_centers, det_radius,
                det_angles in bbox_list
            ]

            bbox_results = results[0][0]
            mask_results = results[0][1]
            center_results = results[0][2]
            radius_results = results[0][3]
            angle_results = results[0][4]

            if show:
                return bbox_results, mask_results, center_results, radius_results, angle_results
            else:
                return bbox_results, mask_results
        elif len(bbox_list[0]) == 4:
            results = [
                vis_bbox_mask2result_xy(det_bboxes, det_masks, det_labels,
                                        det_centers,
                                        self.bbox_head.num_classes,
                                        img_meta[0])
                for det_bboxes, det_labels, det_masks, det_centers in bbox_list
            ]

            bbox_results = results[0][0]
            mask_results = results[0][1]
            xy_results = results[0][2]
            center_results = results[0][3]

            if show:
                return bbox_results, mask_results, xy_results, center_results
            else:
                return bbox_results, mask_results

        # elif len(bbox_list[0]) == 3:

        #     results = [
        #         vis_bbox_mask2result_xy(det_bboxes, det_masks, det_labels,
        #                                 self.bbox_head.num_classes,
        #                                 img_meta[0])
        #         for det_bboxes, det_labels, det_masks in bbox_list
        #     ]

        #     bbox_results = results[0][0]
        #     mask_results = results[0][1]
        #     xy_results = results[0][2]
        #     if show:
        #         return bbox_results, mask_results, xy_results
        #     else:
        #         return bbox_results, mask_results


# for visualize the maked gt in the training process
# set batch_size=1 first
# def forward_train(self,
#                   img,
#                   img_metas,
#                   gt_bboxes,
#                   gt_labels,
#                   gt_masks=None,
#                   gt_bboxes_ignore=None,
#                   _gt_labels=None,
#                   _gt_bboxes=None,
#                   _gt_masks=None,
#                   _gt_contour_all_points=None):
#     if _gt_labels is not None:
#         extra_data = dict(_gt_labels=_gt_labels,
#                           _gt_bboxes=_gt_bboxes,
#                           _gt_masks=_gt_masks,
#                           _gt_contour_all_points=_gt_contour_all_points)
#     else:
#         extra_data = None

#     x = self.extract_feat(
#         img
#     )  # (2,256,96,160) (2,256,48,80) (2,256,24,40) (2,256,12,20) (2,256,6,10)
#     outs = self.bbox_head(x)
#     loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)

#     losses, pos_points, pos_mask_targets, pos_deviation_preds, pos_contour_all_points, pos_inds = self.bbox_head.loss(
#         *loss_inputs,
#         gt_masks=gt_masks,
#         gt_bboxes_ignore=gt_bboxes_ignore,
#         extra_data=extra_data,
#         debug=True)

#     contours = distance2mask(
#         pos_points, pos_mask_targets,
#         pos_deviation_preds.detach() +
#         torch.ones(pos_deviation_preds.shape, requires_grad=False).cuda() *
#         math.pi / 18)

#     # contours = distance2mask(
#     #     pos_points, pos_contour_all_points,
#     #     torch.ones(pos_contour_all_points.shape,
#     #                requires_grad=False).cuda() * math.pi / 18 / 2)

#     # contours = distance2mask(
#     #     pos_points, _gt_masks[0][pos_inds],
#     #     torch.ones(pos_deviation_preds.shape, requires_grad=False).cuda() *
#     #     math.pi / 18)

#     train_img_array = img[0].permute(1, 2, 0).cpu()
#     train_img_array = train_img_array.numpy().astype('uint8')
#     cv2.imwrite(
#         os.path.join('test_gt', 'train_' + img_metas[0]['file_name']),
#         train_img_array)

#     img_array = mmcv.imread('../coco/train2017/' +
#                             img_metas[0]['file_name'])
#     img_array, _, _ = mmcv.imresize(img_array, (1280, 768),
#                                     return_scale=True)

#     if img_metas[0]['flip']:
#         img_array = mmcv.imflip(img_array)
#     cv2.imwrite(os.path.join('test_gt', img_metas[0]['file_name']),
#                 img_array)

#     pos_points = pos_points.cpu().numpy().astype(int)
#     for i in range(contours.shape[0]):
#         img_draw = img_array.copy()
#         im_mask = np.zeros(img_draw.shape[:2], dtype=np.uint8)
#         mask = contours[i].detach().cpu().numpy().astype(
#             int).T[:, None, :]  # convert to int
#         img_draw = cv2.drawContours(img_draw, mask, -1, (0, 0, 255), 10)
#         # im_mask = cv2.drawContours(im_mask, mask, -1, 1, -1)
#         # rle = mask_util.encode(
#         #     np.array(im_mask[:, :, np.newaxis], order='F'))[0]
#         # mask = mask_util.decode(rle).astype(np.bool)
#         # color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
#         # img_draw[mask] = img_draw[mask] * 0.5 + color_mask * 0.5

#         # mask = cv2.fillPoly(
#         #     np.zeros(img_draw.shape[:2]),
#         #     [contours[i].detach().cpu().numpy().astype(int).T],
#         #     (255, 255, 255)).astype(np.bool)
#         #img_draw[mask] = img_draw[mask] * 0.5 + color_mask * 0.5
#         # contour = contours[i].detach().cpu().numpy().astype(int).T
#         # for j in range(contour.shape[0] - 1):
#         #     img_draw = cv2.line(img_draw, tuple(contour[j]),
#         #                         tuple(contour[j + 1]), (0, 0, 255))

#         img_draw = cv2.circle(
#             img_draw,
#             (pos_points[i][0], pos_points[i][1]),
#             5,
#             (0, 0, 255),
#         )
#         cv2.imwrite(
#             os.path.join(
#                 'test_gt', 'img{}_make_gt_{}.jpg'.format(
#                     img_metas[0]['file_name'][:-4], i)), img_draw)
#     import pdb
#     pdb.set_trace()
#     return losses

# for visualize the maked gt in the training process
# def distance2mask(points, distances, angles, max_shape=None):
#     '''Decode distance prediction to 36 mask points
#     Args:
#         points (Tensor): Shape (n, 2), [x, y].
#         distance (Tensor): Distance from the given point to 36,from angle 0 to 350.
#         angles (Tensor): 预测的delta theta
#         max_shape (tuple): Shape of the image.

#     Returns:
#         Tensor: Decoded masks.
#     '''
#     num_points = points.shape[0]
#     points = points[:, :, None].repeat(1, 1, distances.shape[1])
#     c_x, c_y = points[:, 0], points[:, 1]

#     angles = torch.cumsum(angles, dim=1)  # Angle expressed in radians
#     angles = angles.clamp(min=0, max=2 * math.pi)

#     # sin = torch.sin(angles)
#     # cos = torch.cos(angles)
#     # sin = sin[None, :].repeat(num_points, 1)
#     # cos = cos[None, :].repeat(num_points, 1)
#     sin = torch.sin(angles)
#     cos = torch.cos(angles)

#     # for test continuous
#     # import pdb
#     # pdb.set_trace()
#     #distances = distances.mean(1)[:, None].repeat(1, 36)

#     x = distances * sin + c_x
#     y = distances * cos + c_y

#     if max_shape is not None:
#         x = x.clamp(min=0, max=max_shape[1] - 1)
#         y = y.clamp(min=0, max=max_shape[0] - 1)

#     res = torch.cat([x[:, None, :], y[:, None, :]], dim=1)

#     return res
