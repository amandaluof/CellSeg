from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn as nn

from mmdet.core import bbox2result, bbox_mask2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from IPython import embed
import time
import torch
import numpy as np
import cv2
import pycocotools.mask as mask_util

def vis_bbox_mask2result(bboxes, masks, labels, centers, num_classes,
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


@DETECTORS.register_module
class PolarMask(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PolarMask, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      _gt_labels=None,
                      _gt_bboxes=None,
                      _gt_masks=None
                      ):

        if _gt_labels is not None:
            extra_data = dict(_gt_labels=_gt_labels,
                              _gt_bboxes=_gt_bboxes,
                              _gt_masks=_gt_masks)
        else:
            extra_data = None


        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)

        losses = self.bbox_head.loss(
            *loss_inputs,
            gt_masks = gt_masks,
            gt_bboxes_ignore=gt_bboxes_ignore,
            extra_data=extra_data
        )
        return losses


    def simple_test(self, img, img_meta, rescale=False, show=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        if not show:
            results = [
                bbox_mask2result(det_bboxes, det_masks, det_labels, self.bbox_head.num_classes, img_meta[0])
                for det_bboxes, det_labels, det_masks in bbox_list]

            bbox_results = results[0][0]
            mask_results = results[0][1]

            return bbox_results, mask_results
        
        else:
            results = [
            vis_bbox_mask2result(det_bboxes, det_masks, det_labels,
                                     det_centers, self.bbox_head.num_classes,
                                     img_meta[0])
                for det_bboxes, det_labels, det_masks, det_centers in bbox_list
            ]

            bbox_results = results[0][0]
            mask_results = results[0][1]
            xy_results = results[0][2]
            center_results = results[0][3]

            return bbox_results, mask_results, xy_results, center_results



