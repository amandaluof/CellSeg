from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import math
import Polygon as plg
from tqdm import tqdm

from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS
import os.path as osp
import warnings

import mmcv
import numpy as np
from imagecorruptions import corrupt
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset
import torch

from .extra_aug import ExtraAugmentation
from .registry import DATASETS
from .transforms import (BboxTransform, ImageTransform, MaskTransform,
                         Numpy2Tensor, SegMapTransform, SegmapTransform)
from .utils import random_scale, to_tensor, contour_random_walk
from IPython import embed
import time
from scipy.spatial import distance
import torch.nn.functional as F

INF = 1e8
# add for sampling 360 points
NUM_SAMPLING_POINTS = 360


def get_angle(v1, v2=[0, 0, 100, 0]):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    included_angle = angle2 - angle1
    if included_angle < 0:
        included_angle += 360
    return included_angle


@DATASETS.register_module
class Coco_Seg_Dataset_gt(CustomDataset):
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 corruption=None,
                 corruption_severity=1,
                 skip_img_without_anno=True,
                 test_mode=False,
                 fill_instance=False,
                 center_fill_before=False,
                 num_sample_points=36,
                 fixed_gt=True,
                 sort=False,
                 polar_coordinate=True,
                 ensure_inner=False,
                 sample_point_heatmap=False,
                 heatmap_contour=False,
                 explore_times=0,
                 normalize_factor=1,
                 extreme_point=False,
                 polar_xy=False,
                 heatmap_upsample=False,
                 boarden_boundary_lvl=[]):
        super(Coco_Seg_Dataset_gt, self).__init__(
            ann_file, img_prefix, img_scale, img_norm_cfg, multiscale_mode,
            size_divisor, proposal_file, num_max_proposals, flip_ratio,
            with_mask, with_crowd, with_label, with_semantic_seg, seg_prefix,
            seg_scale_factor, extra_aug, resize_keep_ratio, corruption,
            corruption_severity, skip_img_without_anno, test_mode)
        assert 360 % num_sample_points == 0, '360 is not multiple to sampling interval'

        self.fill_instance = fill_instance
        self.center_fill_before = center_fill_before
        self.num_sample_points = num_sample_points
        self.sort = sort
        self.fixed_gt = fixed_gt  # whether to need to make sure the center is in mask
        self.polar_coordinate = polar_coordinate
        self.ensure_inner = ensure_inner
        self.explore_times = explore_times
        self.normalize_factor = normalize_factor
        self.sample_point_heatmap = sample_point_heatmap  # 计算heatmap分支的gt
        self.heatmap_contour = heatmap_contour  # heatmap分支选的点是否是轮廓上的所有点,或是只有采样点
        self.extreme_point = extreme_point  # 在对轮廓点进行采样时是否包含4个极点
        self.polar_xy = polar_xy  # 是否同时采样极坐标和二维坐标
        self.heatmap_upsample = heatmap_upsample  # 制作的gt是是否为下采样4倍（P3下采样8倍）
        self.boarden_boundary_lvl = boarden_boundary_lvl  # 指定特征层的实例轮廓为P3层上对应的2倍

        if self.fixed_gt:
            print('Ground truth is fixed')
        if self.ensure_inner:
            print('Choose points inside mask as center')

        if self.fill_instance:
            print(
                'If there exist more than one parts in a mask, fill it first')
            if self.center_fill_before:
                print(
                    'The center is the average of coordinates of several parts'
                )
            else:
                print('after filling, calculate the center')
        if self.normalize_factor < 1:
            print('normalize the deviation')
        if self.sample_point_heatmap:
            print('calculate the heatmap for additional cls branch')
        if self.polar_xy:
            print('sample 36 polar points and xy together')
        if self.extreme_point:
            print('Four extreme points are included when sampling')
        if self.heatmap_upsample:
            print('The heatmap gt is upsampled')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.

        self.debug = False

        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []

        if self.debug:
            count = 0
            total = 0
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            # filter bbox < 10
            if self.debug:
                total += 1

            if ann['area'] <= 15 or (
                    w < 10 and h < 10) or self.coco.annToMask(ann).sum() < 15:
                # print('filter, area:{},w:{},h:{}'.format(ann['area'],w,h))
                if self.debug:
                    count += 1
                continue

            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)

        if self.debug:
            print('filter:', count / total)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(bboxes=gt_bboxes,
                   labels=gt_labels,
                   bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def prepare_train_img(self, idx):
        '''
        对不含bbox的图像直接返回空；然后对图像进行预处理
        输出：dataset字典，dataset[i]包含键值
        'img':预处理后的图像, DataContainer, img.data: tensor(3,768,1280)
        'img_meta': 原始图像、预处理图像信息，DataContainer, img_meta.data
        'gt_bboxes': 图中实例的bbounding box，DataContainer, gt_bboxes.data: tensor(k, 4)
        'gt_labels': 图中每个实例的下标
        'gt_masks': 图中每个实例的mask, DataContainer, gt_labels.data: tensor(k,768,1280)
        '_gt_labels': 不同尺度的特征图中每个像素对应的标签，DataContainer,_gt_labels.data: tensor(768*1280*(1/64+1/256+1/32/32+1/64/64+1/128/128)),即20460
        '_gt_bboxes'：不同尺度的特征图中每个像素对应的bbox，DataContainer,_gt_bboxes.data: tensor(20460,4)
        '_gt_masks'：不同尺度的特征图中每个像素对应的轮廓点极径，DataContainer,_gt_masks.data: tensor(20460,36)
        '_contour_all_points'：不同尺度的特征图中每个像素对应的360个轮廓点的极径，DataContainer,_contour_all_points.data: tensor(20460,360)
        '''

        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # corruption
        if self.corruption is not None:
            img = corrupt(img,
                          severity=self.corruption_severity,
                          corruption_name=self.corruption)
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)

        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0 and self.skip_img_without_anno:
            warnings.warn('Skip the image "%s" that has no valid gt bbox' %
                          osp.join(self.img_prefix, img_info['filename']))
            return None

        # apply transforms，比如：resacle、归一化、翻转、pad、通道转置
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)

        img = img.copy()
        if self.with_seg:
            gt_seg = mmcv.imread(osp.join(
                self.seg_prefix, img_info['filename'].replace('jpg', 'png')),
                                 flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(gt_seg,
                                    self.seg_scale_factor,
                                    interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack([proposals, scores
                                   ]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(file_name=img_info['filename'],
                        ori_shape=ori_shape,
                        img_shape=img_shape,
                        pad_shape=pad_shape,
                        scale_factor=scale_factor,
                        flip=flip)

        data = dict(img=DC(to_tensor(img), stack=True),
                    img_meta=DC(img_meta, cpu_only=True),
                    gt_bboxes=DC(to_tensor(gt_bboxes)))

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)

        # --------------------offline ray label generation-----------------------------

        self.center_sample = True
        self.use_mask_center = True
        self.radius = 1.5
        self.strides = [8, 16, 32, 64, 128]
        self.regress_ranges = (
            (-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)
        )  # 限定不同特征层的像素回归极径的值的范围不同，低级的层（尺寸最大）最小
        featmap_sizes = self.get_featmap_size(pad_shape)
        self.featmap_sizes = featmap_sizes
        num_levels = len(self.strides)
        all_level_points = self.get_points(featmap_sizes)  # 所有特征点对应到原图的坐标
        self.num_points_per_level = [i.size()[0] for i in all_level_points]

        expanded_regress_ranges = [
            all_level_points[i].new_tensor(
                self.regress_ranges[i])[None].expand_as(all_level_points[i])
            for i in range(num_levels)
        ]  # 将这个回归的范围限制赋给了特征图每一层的每个像素
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(all_level_points, 0)
        gt_masks = gt_masks[:len(gt_bboxes)]

        gt_bboxes = torch.Tensor(gt_bboxes)
        gt_labels = torch.Tensor(gt_labels)

        # 获取对应的极径极角,或是对应的二维坐标
        if self.polar_coordinate:
            _labels, _bbox_targets, _mask_targets, _angle_targets = self.polar_target_single(
                gt_bboxes, gt_masks, gt_labels, concat_points,
                concat_regress_ranges)
        else:
            # _labels, _bbox_targets, _mask_targets, _gt_sample_points_id, _points_sample_gt_dict = self.polar_target_single(
            #     gt_bboxes, gt_masks, gt_labels, concat_points,
            #     concat_regress_ranges)
            if self.sample_point_heatmap:
                _labels, _bbox_targets, _mask_targets, _gt_sample_heatmap = self.polar_target_single(
                    gt_bboxes, gt_masks, gt_labels, concat_points,
                    concat_regress_ranges)
            elif self.polar_xy:
                _labels, _bbox_targets, _mask_targets, _polar_targets = self.polar_target_single(
                    gt_bboxes, gt_masks, gt_labels, concat_points,
                    concat_regress_ranges)
            else:
                _labels, _bbox_targets, _mask_targets = self.polar_target_single(
                    gt_bboxes, gt_masks, gt_labels, concat_points,
                    concat_regress_ranges)

        data['_gt_labels'] = DC(_labels)
        data['_gt_bboxes'] = DC(_bbox_targets)

        if self.polar_coordinate:
            data['_gt_radius'] = DC(_mask_targets)
            data['_gt_angles'] = DC(_angle_targets)
        else:
            data['_gt_xy_targets'] = DC(_mask_targets)  # (num, 36, 2)
            if self.polar_xy:
                data['_gt_polar_targets'] = DC(_polar_targets)
            if self.sample_point_heatmap:
                data['_gt_sample_heatmap'] = DC(
                    _gt_sample_heatmap)  # 尺寸：(5,H/8,W/8)，每一层采样点，绘制到(96,160)

        # 可视化mask
        # cv2.imwrite('original.jpg',
        #             data['gt_masks'].data.sum(0).clip(0, 1) * 255)
        # --------------------offline ray label generation-----------------------------
        return data

    def get_featmap_size(self, shape):
        h, w = shape[:2]
        featmap_sizes = []
        for i in self.strides:
            featmap_sizes.append([int(h / i), int(w / i)])
        return featmap_sizes

    def get_points(self, featmap_sizes):
        '''
        得到指定尺度范围的所有特征图在原图对应的坐标 (xs+s//2, ys+s//2)
        '''
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i]))
        return mlvl_points

    def get_points_single(self, featmap_size, stride):
        '''
        得到传入尺寸的feature map在原图对应的坐标 (xs+s//2, ys+s//2)
        '''
        h, w = featmap_size
        x_range = torch.arange(0, w * stride, stride)
        y_range = torch.arange(0, h * stride, stride)
        y, x = torch.meshgrid(y_range, x_range)  # 将x和y组成n^2个数
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)),
            dim=-1) + stride // 2  # (xs+s//2, ys+s//2)是最靠近像素对应的区域的中心
        return points.float()  # 是反的

    def polar_target_single(self, gt_bboxes, gt_masks, gt_labels, points,
                            regress_ranges):
        '''
        对feature map上的每个点确定用极坐标表示的bbox和轮廓点

        input:
        gt_bboxes: 表示图中所有实例的bbox shape (k,4) (x1,y1,x2,y2)
        gt_masks: shape (k,768,1280)
        gt_labels: shape (k)
        points: 所有特征图上所有的点(对应到原图的位置)
        regress_ranges: 所有特征图上所有点（对应回原图的坐标）的回归值的范围
        output:
        labels: 每个点对应回原图的未知的类别（0为背景）
        bbox_taegets: 每个像素对应回原图的点距离其bbox的四条边的距离（bbox是按照mask的assign方法找的）
        mask_targets: 每个像素对应回原图的点如果在某bbox内或者在回归范围内，则将实例指定到该点
        '''
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                gt_bboxes.new_zeros((num_points, 4))

        # areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] -
        #                                                    gt_bboxes[:, 1] + 1)
        # # TODO: figure out why these two are different
        # # areas = areas[None].expand(num_points, num_gts)
        # areas = areas[None].repeat(num_points, 1)  # [None]升维操作
        # regress_ranges = regress_ranges[:, None, :].expand(
        #     num_points, num_gts, 2)  # 将原有的回归范围限制又复制了实例个数次
        # gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        # # xs ys 分别是points的x y坐标
        # xs, ys = points[:, 0], points[:, 1]
        # xs = xs[:, None].expand(num_points, num_gts)
        # ys = ys[:, None].expand(num_points, num_gts)
        # left = xs - gt_bboxes[..., 0]
        # right = gt_bboxes[..., 2] - xs
        # top = ys - gt_bboxes[..., 1]
        # bottom = gt_bboxes[..., 3] - ys
        # # feature map上所有点对于gtbox的上下左右距离 [num_pix, num_gt, 4]
        # bbox_targets = torch.stack((left, top, right, bottom), -1)

        # mask targets 也按照这种写 同时labels 得从bbox中心修改成mask 重心
        mask_centers = []
        mask_contours = []
        instance_used_idx = [
        ]  # 记录所有使用的instance，在固定gt时，有些实例因为太小找不到内部的中心点没有参与计算
        # 第一步 先算k个重心  return [num_gt, 2]
        for i in range(gt_masks.shape[0]):
            mask = gt_masks[i]
            cnt, contour = self.get_single_centerpoint(mask)  # cnt应该与cv2中的顺序一致
            if cnt is None:
                print('skip a vey small mask')
                continue
            contour = contour[0]  # contour with the largest area
            contour = torch.Tensor(contour).float()
            y, x = cnt
            mask_centers.append([x, y])  # mask center是正的
            mask_contours.append(contour)
            instance_used_idx.append(i)
        # 在固定gt时，会依据mask中心是否可以在mask内部，滤除掉不符合条件的实例
        # 然后先计算bbox gt，再计算mask gt
        num_gts = len(instance_used_idx)
        gt_bboxes, gt_labels = gt_bboxes[instance_used_idx], gt_labels[
            instance_used_idx]
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] -
                                                           gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)  # [None]升维操作
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)  # 将原有的回归范围限制又复制了实例个数次
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        # xs ys 分别是points的x y坐标
        xs, ys = points[:, 0], points[:, 1]  # points是反的
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        # feature map上所有点对于gtbox的上下左右距离 [num_pix, num_gt, 4]
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # 分配实例到像素，制作_gt_mask
        mask_centers = torch.Tensor(mask_centers).float()
        # 把mask_centers assign到不同的层上,根据regress_range和重心的位置
        mask_centers = mask_centers[None].expand(
            num_points, num_gts, 2)  # 每个特征点对应原图像素都可能是num_gts个实例中的一个

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------
        # condition1: inside a gt bbox
        # 加入center sample
        if self.center_sample:
            strides = [8, 16, 32, 64, 128]
            if self.use_mask_center:
                inside_gt_bbox_mask = self.get_mask_sample_region(
                    gt_bboxes,
                    mask_centers,
                    strides,
                    self.num_points_per_level,
                    xs,
                    ys,
                    radius=self.radius)
            else:
                inside_gt_bbox_mask = self.get_sample_region(
                    gt_bboxes,
                    strides,
                    self.num_points_per_level,
                    xs,
                    ys,
                    radius=self.radius)
        else:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]  # 该点离bbox最远边界的距离

        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])  # 最大回归距离在回归范围内

        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(
            dim=1)  # 如果属于多个实例取面积最小的那个，不属于实例的面积会取INF

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0  # [num_gt] 介于0-80,背景像素为0

        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        pos_inds = labels.nonzero().reshape(-1)  # positive examples index

        if self.polar_coordinate:
            angle_targets = torch.zeros(num_points, 36).float()
            radius_targets = torch.zeros(num_points, 36).float()
        else:
            points_targets = torch.zeros(num_points, 36, 2).float()
            mask_id_targets = torch.ones(num_points, 1) * (-1)
            if self.polar_xy:
                polar_targets = torch.zeros(num_points, 36).float()

        pos_mask_ids = min_area_inds[pos_inds]

        # 在所有实例选相同点时，对于后面重复使用的mask仅等间隔采样一次轮廓点
        if not self.polar_xy:
            pos_mask_ids_set = list(set(pos_mask_ids.data.numpy()))
            points_sample_gt_dict = dict({})
            for mask_id in pos_mask_ids_set:
                mask_contour = mask_contours[mask_id][:, 0, :]
                mask = gt_masks[mask_id]
                points_sample_gt = contour_random_walk(
                    num_sample_points=self.num_sample_points,
                    all_contour_points=mask_contour,
                    mask=mask,
                    explore_times=self.explore_times,
                    num_steps=10,
                    step=1,
                    show_iou=False,
                    extreme_included=self.extreme_point)
                points_sample_gt_dict[mask_id] = points_sample_gt

        for p, id in zip(pos_inds, pos_mask_ids):
            x, y = points[p]
            # 采出等间隔的36个点的极径
            if self.polar_xy:
                mask_contour = mask_contours[id]
                dists, _, start = self.get_36_coordinates(
                    gt_masks[id], x, y,
                    mask_contour)  # 此处得到的点对应的极角是[180，...,350,0,...170]
                polar_targets[p] = dists

                ####################
                # 找到起点在轮廓点中的序号，并以此为起点逆时针等间隔采点
                mask_contour = mask_contour[:, 0, :]  # (n,2)
                start_idx_bool = ((mask_contour == start).sum(1) == 2)
                # polar采样的点在轮廓点上序号
                start_idx = int(
                    torch.range(
                        0,
                        len(mask_contour) -
                        1)[start_idx_bool].numpy().tolist()[0])  # 轮廓中会出现两个一样的点
                # print(start)
                # print(mask_contour[start_idx])
                points_sample_gt = contour_random_walk(
                    num_sample_points=self.num_sample_points,
                    all_contour_points=mask_contour,
                    mask=gt_masks[id],
                    explore_times=self.explore_times,
                    num_steps=10,
                    step=1,
                    start=start_idx,
                    show_iou=False,
                    extreme_included=self.extreme_point)
                points_targets[p] = self.get_deviation_xy(
                    x, y, points_sample_gt)

            else:
                mask_id = id.numpy().tolist()
                points_sample_gt = points_sample_gt_dict[mask_id]
                mask_id_targets[p] = mask_id  # 保存每个实例对应的mask_id,方便后面取每一层的轮廓采样点
                # print('gt iou:', iou)
                if self.polar_coordinate:
                    _, angle_targets[p], radius_targets[
                        p] = self.get_angle_radius(x, y, points_sample_gt)
                else:
                    points_targets[p] = self.get_deviation_xy(
                        x, y, points_sample_gt)  # 仍然是y,x的形式
                    # 对每个正样本都会等间隔计算一次

                # 可视化
                # angles = torch.cat(
                #     [torch.range(180, 350, 10),
                #      torch.range(0, 170, 10)]) / 180 * math.pi  #
                # polar_points = distance2mask(points[p].reshape(1,
                #                                                2), dists, angles,
                #                              mask.shape).permute(0, 2,
                #                                                  1)  # (1,36,2)
                # mask = gt_masks[id]
                # mask = mask * 255
                # mask[int(y.numpy().tolist()), int(x.numpy().tolist())] = 0
                # mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]],axis=2)

                # mask[points_sample_gt[:, 1].int().numpy().tolist(), points_sample_gt[:, 0].int().numpy().tolist(), :2] = 0
                # mask = cv2.polylines(
                #     mask,
                #     points_sample_gt.numpy().astype(np.int32)[1:10][None, :, :],
                #     True, (0, 0, 255), 2)

                # mask[polar_points[0, :, 1].int().numpy().tolist(),
                #      polar_points[0, :, 0].int().numpy().tolist(), 2] = 0
                # polar_points = polar_points.permute(1, 0, 2)
                # import pdb
                # pdb.set_trace()

                #####
                # mask = cv2.polylines(
                #     mask,
                #     polar_points.numpy().astype(np.int32)[:, :1, :], True,
                #     (0, 255, 0), 2)
                # cv2.imwrite('demo.jpg', mask)
                # import pdb
                # pdb.set_trace()

        # 把每一层实例的所有点绘制到H*W的map上 (5,H,W)
        # 所有像素分层，对每层像素所属的mask求集合，然后去除对应的采样点，并将其绘制到(96,160)的画布上
        # TODO: 对于在poalr_xy的情况下的heatmap暂时没有实现
        if self.sample_point_heatmap:
            lvl_mask_ids = mask_id_targets.split([15360, 3840, 960, 240, 60],
                                                 0)
            lvl_mask_ids = [
                list(set(mask_id[:, 0].cpu().numpy().tolist()) - set({-1}))
                for mask_id in lvl_mask_ids
            ]
            #points_sample_map_list = [] # 在原图上绘制的点
            points_sample_map_rescale_list = []
            for i in range(5):
                # mask_points = torch.zeros(mask.shape) # 原图大小
                if self.heatmap_upsample:
                    mask_points_rescale = torch.zeros(
                        (int(mask.shape[0] / 4),
                         int(mask.shape[1] / 4)))  # 制作的heatmap gt是下采样4倍的结果

                else:
                    mask_points_rescale = torch.zeros(
                        (int(mask.shape[0] / 8), int(mask.shape[1] / 8)
                         ))  # 针对P3，确定生成的分类标签的图尺寸为下采样8倍 # TODO：如果要修改在其他尺寸上绘图

                if not self.heatmap_contour:
                    all_points = [
                        points_sample_gt_dict[lvl_mask_ids[i][j]]
                        for j in range(len(lvl_mask_ids[i]))
                    ]
                else:
                    all_points = [
                        mask_contours[int(lvl_mask_ids[i][j])][:, 0, :]
                        for j in range(len(lvl_mask_ids[i]))
                    ]
                if len(all_points) > 0:
                    all_points = torch.cat(all_points).long()
                    # mask_points[all_points[:, 1],
                    #             all_points[:, 0]] = 1  # 在原图上绘制点，点的坐标是按照cv2取的
                    if self.heatmap_upsample:
                        all_points_rescale = (all_points / 4).long()
                    else:
                        all_points_rescale = (all_points / 8).long()
                    if i in self.boarden_boundary_lvl:
                        mask_points_rescale = torch.zeros(
                            (int(mask.shape[0] / 8), int(mask.shape[1] / 8)))
                        all_points_rescale = (all_points / 8).long()
                        mask_points_rescale[all_points_rescale[:, 1],
                                            all_points_rescale[:, 0]] = 1
                        mask_points_rescale = F.interpolate(
                            mask_points_rescale[None, None, :, :],
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False)[0, 0, :, :]
                        mask_points_rescale = (mask_points_rescale >
                                               0.5).float()  # 若为0，则轮廓会再宽2倍

                    else:
                        mask_points_rescale[all_points_rescale[:, 1],
                                            all_points_rescale[:, 0]] = 1

                points_sample_map_rescale_list.append(
                    mask_points_rescale[None, :, :])  #在下采样8倍的图大小上绘制的点
                # points_sample_map_list.append(
                #     mask_points[None, :, :])  #保存原始图大小上绘制的点

            # points_sample_map_lvl = torch.cat(points_sample_map_list, 0) # 在原图上绘制的点
            points_sample_map_rescale_lvl = torch.cat(
                points_sample_map_rescale_list, 0)

        # # 可视化在原图上绘制采的点，在下采样8倍上绘制采的点，各个实例的mask
        # for i in range(5):
        #     # sample_map = points_sample_map_lvl[i].numpy().astype(np.uint8)
        #     sample_map_rescale = points_sample_map_rescale_lvl[i].numpy(
        #     ).astype(np.uint8)
        #     # cv2.imwrite('d_{}.jpg'.format(i), sample_map * 255)
        #     cv2.imwrite('d_{}_rescale.jpg'.format(i), sample_map_rescale * 255)

        # for i in range(len(gt_masks)):
        #     cv2.imwrite('dd_{}.jpg'.format(i), gt_masks[i] * 255)

        # 可视化
        # for p, id in zip(pos_inds, pos_mask_ids):
        #     x, y = points[p]
        #     pos_mask_contour = mask_contours[id]

        #     # 对每个实例求点和坐标
        #     mask = gt_masks[instance_used_idx][id]
        #     import pdb
        #     pdb.set_trace()
        #     ct = pos_mask_contour[:, 0, :]
        #     points_sample_gt = contour_random_walk(
        #         num_sample_points=self.num_sample_points,
        #         all_contour_points=ct,
        #         mask=mask,
        #         explore_times=self.explore_times,
        #         num_steps=10,
        #         step=1,
        #         show_iou=False)

        # # print('gt iou:', iou)
        # if self.polar_coordinate:
        #     _, angle_targets[p], radius_targets[p] = self.get_angle_radius(
        #         x, y, points_sample_gt)
        # else:
        #     points_targets[p] = self.get_deviation_xy(
        #         x, y, points_sample_gt)  # 仍然是y,x的形式

        # from .utils import points_mask_iou
        # print(points_mask_iou(points_sample_gt, mask))

        # 可视化gt
        # mask = gt_masks[instance_used_idx][id]
        # mask = mask * 255
        # mask = np.concatenate(
        #     [mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
        # mask = cv2.circle(
        #     mask, (x, y), 1,
        #     (0, 0, 255))  # 当前像素,正样本像素不一定在物体内，在物体很小的情况下，可能在物体外部(1.5stride）
        # mask = cv2.circle(mask,
        #                   (mask_centers[p][id][1], mask_centers[p][id][0]),
        #                   1, (255, 0, 0))  # real mass center

        # mask = cv2.polylines(
        #     mask,
        #     points_sample_gt.numpy().astype(np.int32)[1:10][None, :, :],
        #     True, (0, 0, 255), 2)
        # mask = cv2.polylines(
        #     mask,
        #     points_sample_gt.numpy().astype(np.int32)[10:][None, :, :],
        #     True, (255, 0, 0), 2)

        # mask = cv2.polylines(
        #     mask,
        #     points_sample_gt.numpy().astype(np.int32)[:1][:, None, :],
        #     True, (0, 255, 0), 2)
        # cv2.imwrite('demo.jpg'.format(id), mask)
        # import pdb
        # pdb.set_trace()

        #test for polar2cartesian
        # from mmdet.models.anchor_heads.polarmask_double_gt_head import distance2mask
        # import pdb
        # pdb.set_trace()
        # a = distance2mask(torch.Tensor([[x, y]]), radius_targets[p],
        #                   angle_targets[p])
        # print(a[0,:,:] == points_sample_gt.T)

        if self.polar_coordinate:
            return labels, bbox_targets, radius_targets, angle_targets
        else:
            if self.polar_xy:
                return labels, bbox_targets, points_targets, polar_targets
            if self.sample_point_heatmap:
                return labels, bbox_targets, points_targets, points_sample_map_rescale_lvl  # (5,36,2)
            else:
                return labels, bbox_targets, points_targets  # (36,2)

    def get_sample_region(self,
                          gt,
                          strides,
                          num_points_per,
                          gt_xs,
                          gt_ys,
                          radius=1):
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)

        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0],
                                                   xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1],
                                                   ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2],
                                                   gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3],
                                                   gt[beg:end, :, 3], ymax)
            beg = end

        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  # 上下左右都>0 就是在bbox里面
        return inside_gt_bbox_mask

    def get_mask_sample_region(self,
                               gt_bb,
                               mask_center,
                               strides,
                               num_points_per,
                               gt_xs,
                               gt_ys,
                               radius=1):
        '''
        判断像是否为中心像素，需要满足的条件:像素在实际质心的正负1.5倍步长的范围内，并且像素在bbox内
        input:
        mask_center:(num_pixel, num_instance,2),传入是x，y
        (gt_xs, gt_ys):图像中点的横纵坐标，需要判断点是否是中心像素
        '''
        center_y = mask_center[..., 0]
        center_x = mask_center[..., 1]
        center_gt = gt_bb.new_zeros(gt_bb.shape)  # (num_pixel, num_instance,2)
        # no gt,该图像不含实例
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)

        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[
                beg:end] + stride  # (num_points_per_vl,num_instance)
            ymax = center_y[beg:end] + stride
            # limit sample region in gt bbox
            center_gt[beg:end, :, 0] = torch.where(xmin > gt_bb[beg:end, :, 0],
                                                   xmin, gt_bb[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt_bb[beg:end, :, 1],
                                                   ymin, gt_bb[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt_bb[beg:end, :, 2],
                                                   gt_bb[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt_bb[beg:end, :, 3],
                                                   gt_bb[beg:end, :, 3], ymax)
            beg = end

        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  # 上下左右都>0 就是在bbox里面
        return inside_gt_bbox_mask

    def get_centerpoint(self, lis):
        '''
        https://www.jianshu.com/p/77234d0ac24e
        多边形质心求解，是三角剖分后的各三角形的质心的加权和
        '''
        area = 0.0
        x, y = 0.0, 0.0
        a = len(lis)
        for i in range(a):
            lat = lis[i][0]
            lng = lis[i][1]
            if i == 0:
                lat1 = lis[-1][0]
                lng1 = lis[-1][1]
            else:
                lat1 = lis[i - 1][0]
                lng1 = lis[i - 1][1]
            fg = (lat * lng1 - lng * lat1) / 2.0
            area += fg
            x += fg * (lat + lat1) / 3.0
            y += fg * (lng + lng1) / 3.0
        x = x / area
        y = y / area

        return [int(x), int(y)]

    def inner_dot(self, instance_mask, point):
        '''
        point是cv2中的格式(h,w)
        instance与point一致
        '''
        xp, yp = point
        h, w = instance_mask.shape
        bool_inst_mask = instance_mask.astype(bool)
        neg_bool_inst_mask = 1 - bool_inst_mask  # 背景像素
        dot_mask = np.zeros(instance_mask.shape)
        insth, instw = instance_mask.shape
        try:
            dot_mask[yp][xp] = 1
        except:
            import pdb
            pdb.set_trace()
        # point不在instance mask内部
        if yp + 1 >= h or yp - 1 < 0 or xp + 1 >= w or xp - 1 < 0:
            return False
        fill_mask = np.zeros((3, 3))
        fill_mask.fill(1)

        dot_mask[yp - 1:yp + 2,
                 xp - 1:xp + 2] = fill_mask  # 中心点及中心点附近的元素 3*3,说明中心点在轮廓上
        # 背景像素中存在在中心点附近的像素 ——> 中心点附近存在背景像素
        not_inner = (neg_bool_inst_mask * dot_mask).any()
        # print(np.sum(neg_bool_inst_mask),np.sum(dot_mask))
        # print('neg_bool',np.unique(dot_mask))
        return not not_inner

    def judge_inner(self, mask, center):
        '''
        center: shape (2) 与cv2一致
        '''
        bool_inst_mask = mask.astype(bool)
        temp = np.zeros(mask.shape)
        temp[int(center[1])][int(center[0])] = 1  # 数组和绘图使用的点是反的 /同的
        if (bool_inst_mask * temp).any() and self.inner_dot(mask, center):
            return True
        else:
            return False

    def add_edge(self, im):
        h, w = im.shape[0], im.shape[1]
        add_edge_im = np.zeros((h + 10, w + 10))
        add_edge_im[5:h + 5, 5:w + 5] = im
        add_edge_im = im
        return add_edge_im

    def get_gradient(self, im):
        h, w = im.shape[0], im.shape[1]
        im = self.add_edge(im)  # padding
        instance_id = np.unique(im)[1]
        # delete line
        mask = np.zeros((im.shape[0], im.shape[1]))
        mask.fill(instance_id)
        boolmask = (im == mask)
        im = im * boolmask  # only has object,and use one color to fill the instance

        y = np.gradient(im)[0]
        x = np.gradient(im)[1]
        gradient = abs(x) + abs(y)
        bool_gradient = gradient.astype(bool)
        mask.fill(1)
        gradient_map = mask * bool_gradient * boolmask
        gradient_map = gradient_map[5:h + 5, 5:w + 5]
        # 2d gradient map
        return gradient_map

    def get_inner_center(self, instance_mask, bounding_order):
        '''     
        输出坐标与mask的xy一致
        count (N,2) 与cv2.contour输出顺序一致
        '''
        inst_mask_h, inst_mask_w = np.where(
            instance_mask)  # coordinates for foreground pixels

        # # get gradient_map
        # gradient_map = self.get_gradient(instance_mask)
        # grad_h, grad_w = np.where(gradient_map == 1)  # coordinates for contour

        # inst_points
        inst_points = np.array([[inst_mask_w[i], inst_mask_h[i]]
                                for i in range(len(inst_mask_h))])
        # # edge_points
        # bounding_order = np.array([[grad_w[i], grad_h[i]]
        #                            for i in range(len(grad_h))])
        flag = True
        compact_factor = 1
        while flag:
            try:
                distance_result = distance.cdist(inst_points[::compact_factor],
                                                 bounding_order, 'euclidean')
                flag = False
            except:
                compact_factor = compact_factor * 2

        sum_distance = np.sum(distance_result, 1)
        center_index = np.argmin(sum_distance)

        center_distance = (inst_points[center_index][0],
                           inst_points[center_index][1])  # 与cv2一致
        times_num = 0
        while not self.inner_dot(instance_mask, center_distance):
            times_num += 1
            sum_distance = np.delete(sum_distance, center_index)
            if len(sum_distance) == 0:
                print('no center, return a fake center')
                return None

            center_index = np.argmin(sum_distance)
            center_distance = [
                int(inst_points[center_index][0]),
                int(inst_points[center_index][1])
            ]  # (h, w)
        # import pdb
        # pdb.set_trace()
        # a = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3))
        # cv2.polylines(a, bounding_order[:, None, :], True, (0, 0, 255), 2)
        # cv2.imwrite('b.jpg', a)
        return center_distance

    def get_single_centerpoint(self, mask):
        '''
        input: 掩码
        output: 中心坐标（与轮廓的xy一致）、掩码中的所有轮廓(按照面积大小排序)
        '''
        contour, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_NONE)
        contour.sort(key=lambda x: cv2.contourArea(x),
                     reverse=True)  # only save the biggest one
        '''debug IndexError: list index out of range'''
        count = contour[0][:, 0, :]  # max:1051 590
        try:
            center = self.get_centerpoint(count)  # 将质心作为中心
        except:
            # print('area is 0 when calculate mass center')
            x, y = count.mean(axis=0)
            center = [int(x), int(y)]  # 直接求坐标均值,与轮廓点的xy顺序一致
        if self.fixed_gt:
            # judge 'center' is in mask or not，if not, choose a point in mask
            temp = np.concatenate(
                [mask[:, :, None], mask[:, :, None], mask[:, :, None]],
                axis=2) * 255
            # cv2.circle(temp, tuple(center), 5, (0, 0, 255))
            # cv2.imwrite('demo.jpg', temp)
            if self.ensure_inner:
                if not self.judge_inner(mask, center):
                    print('mass center is not in mask')
                    center = self.get_inner_center(mask, count)
                    # cv2.circle(temp, tuple(center), 5, (0, 0, 255))
                    # cv2.imwrite('demo.jpg', temp)
                    import pdb
                    pdb.set_trace()

        return center, contour  # center is reverse

    def get_angle_radius(self, c_x, c_y, target_contour_points_yx):
        '''
        输入mask，按照沿着轮廓点采样的方式得到gt
        '''
        x = target_contour_points_yx[:, 0] - c_x
        y = target_contour_points_yx[:, 1] - c_y
        angle_arc = torch.atan2(x, y)
        radius = torch.sqrt(x**2 + y**2)
        if self.sort:
            print('use fixed gt with order — angle increasing')
            angle_temp = angle_arc * 180 / np.pi
            angle_temp[angle_temp < 0] += 360
            angle_temp, idx = torch.sort(angle_temp)
            radius = radius[idx]
            target_contour_points_yx = target_contour_points_yx[idx]
            angle_arc = angle_temp / 180 * np.pi
        return target_contour_points_yx, angle_arc, radius

    def get_deviation_xy(self, c_x, c_y, target_contour_points_yx):
        '''
        输入mask，按照沿着轮廓点采样的方式得到gt，输出的是当前点距离中心的偏移
        '''
        deviation_x = target_contour_points_yx[:, 0] - c_x
        deviation_y = target_contour_points_yx[:, 1] - c_y
        deviation_x = deviation_x[:, None]
        deviation_y = deviation_y[:, None]
        deviation = torch.cat([deviation_x, deviation_y], dim=1)
        # 在计算损失时再对偏移进行归一化，减少数值误差
        if self.normalize_factor < 1:
            deviation = deviation * self.normalize_factor
        return deviation

    def get_36_coordinates(self, mask, c_x, c_y, pos_mask_contour):
        '''
        input: 极心坐标, 掩码轮廓
        output: 36个点的极坐标表示, 36个点的极径, start(shape:[2])
        找每个角度周围正负3的点, 如果他们在轮廓点中出现过, 则将他们到轮廓点的距离作为极径
        输出的点是从质心的正上方开始逆时针旋转 **对应distance中的角度应该是（180...360,0,...170)
        '''
        ct = pos_mask_contour[:, 0, :]
        x = ct[:, 0] - c_x  # (N,2)
        y = ct[:, 1] - c_y
        # angle = np.arctan2(x, y)*180/np.pi
        angle = torch.atan2(x, y) * 180 / np.pi  # 点的位置就能反应角度,计算极角则可得到
        angle[angle < 0] += 360
        angle = angle.int()
        # dist = np.sqrt(x ** 2 + y ** 2)
        dist = torch.sqrt(x**2 + y**2)
        angle, idx = torch.sort(angle)
        dist = dist[idx]
        ct = ct[idx]  # (N,2)

        # 生成36个角度,需要记录下180度对应的点的坐标
        new_coordinate = {}
        angle_xy = {}
        for i in range(0, 360, 10):
            if i in angle:
                d, ix = dist[angle == i].max(0)
                new_coordinate[i] = d
                angle_xy[i] = ct[angle == i][ix]
            elif i + 1 in angle:
                d, ix = dist[angle == i + 1].max(0)
                new_coordinate[i] = d
                angle_xy[i] = ct[angle == i + 1][ix]
            elif i - 1 in angle:
                d, ix = dist[angle == i - 1].max(0)
                new_coordinate[i] = d
                angle_xy[i] = ct[angle == i - 1][ix]
            elif i + 2 in angle:
                d, ix = dist[angle == i + 2].max(0)
                new_coordinate[i] = d
                angle_xy[i] = ct[angle == i + 2][ix]
            elif i - 2 in angle:
                d, ix = dist[angle == i - 2].max(0)
                new_coordinate[i] = d
                angle_xy[i] = ct[angle == i - 2][ix]
            elif i + 3 in angle:
                d, ix = dist[angle == i + 3].max(0)
                new_coordinate[i] = d
                angle_xy[i] = ct[angle == i + 3][ix]
            elif i - 3 in angle:
                d, ix = dist[angle == i - 3].max(0)
                new_coordinate[i] = d
                angle_xy[i] = ct[angle == i - 3][ix]
            # else:
            #     if i == 180:
            #         start = None
            #         print('start is None')

        distances = torch.zeros(36)
        flag = False
        for a in range(0, 360, 10):
            if not a in new_coordinate.keys():
                new_coordinate[a] = torch.tensor(1e-6)
                distances[a // 10] = 1e-6
                if a == 0:
                    # polar起始点不在轮廓上，找离polar起点最近的点
                    # print('polar point is not on contour')
                    # print(self.judge_inner(mask, np.array([c_x, c_y])))
                    # 可视化
                    # flag = True
                    # mask = np.concatenate(
                    #     [mask[:, :, None], mask[:, :, None], mask[:, :, None]],
                    #     axis=2)
                    # cv2.polylines(mask,
                    #               pos_mask_contour.int().numpy(), True,
                    #               (0, 255, 0), 1)
                    # cv2.circle(mask, (c_x, c_y), 2, (0, 0, 255))  # 当前像素
                    start = None
                    angles = torch.range(0, 350, 10) / 180 * math.pi
                    polar_points_xy = distance2mask(
                        torch.Tensor([c_x, c_y]).reshape(1, 2), distances,
                        angles).permute(0, 2, 1)  # (1,36,2)
                    start = torch.round(polar_points_xy[0,
                                                        0, :])  # Tensor ([2])
                    # cv2.circle(
                    #     mask, (start.int().numpy()[0], start.int().numpy()[1]),
                    #     5, (255, 0, 0))  # 极坐标的起点
                    dist_contour_start_idx = ((
                        ct - start[None, :]
                    )**2).sum(1).argmin().item(
                    )  # find the point that is closest to the start of polar
                    start = ct[dist_contour_start_idx, :]
                    # cv2.circle(
                    #     mask, (start.int().numpy()[0], start.int().numpy()[1]),
                    #     5, (0, 0, 255))  # xy的起点

            else:
                distances[a // 10] = new_coordinate[a]
                if a == 0:
                    start = angle_xy[a]
        # 可视化
        # if flag:
        #     angles = torch.range(0, 350, 10) / 180 * math.pi
        #     polar_points_xy = distance2mask(
        #         torch.Tensor([c_x, c_y]).reshape(1, 2), distances,
        #         angles).permute(0, 2, 1)  # (1,36,2)
        #     cv2.polylines(mask,
        #                   polar_points_xy.int().numpy(), True, (0, 0, 255), 1)
        #     cv2.imwrite('demo.jpg', mask)
        #     import pdb
        #     pdb.set_trace()

        # distances = torch.cat([distances[18:], distances[:18]])
        return distances, new_coordinate, start

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)

            if data is None:
                idx = self._rand_another(idx)
                continue
            return data


def polar2cartesian(center, radius, angle):
    '''
    与distance2mask一致
    '''
    x, y = center
    x = x + torch.sin(angle) * radius
    y = y + torch.cos(angle) * radius
    res = torch.cat([x[:, None, :], y[:, None, :]], dim=1)

    return res


def distance2mask(points, distances, angles, max_shape=None):
    '''Decode distance prediction to 36 mask points
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 36,from angle 0 to 350.
        angles (Tensor):
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded masks.
    '''
    num_points = points.shape[0]
    points = points[:, :, None].repeat(1, 1, 36)
    c_x, c_y = points[:, 0], points[:, 1]

    sin = torch.sin(angles)
    cos = torch.cos(angles)
    sin = sin[None, :].repeat(num_points, 1)
    cos = cos[None, :].repeat(num_points, 1)

    x = distances * sin + c_x
    y = distances * cos + c_y

    if max_shape is not None:
        x = x.clamp(min=0, max=max_shape[1] - 1)
        y = y.clamp(min=0, max=max_shape[0] - 1)

    res = torch.cat([x[:, None, :], y[:, None, :]], dim=1)
    return res