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
from .utils import random_scale, to_tensor
from IPython import embed
import time
from scipy.spatial import distance

INF = 1e8


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
class Coco_Seg_Dataset(CustomDataset):
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
                 center_fill_before=False):
        super(Coco_Seg_Dataset, self).__init__(
            ann_file, img_prefix, img_scale, img_norm_cfg, multiscale_mode,
            size_divisor, proposal_file, num_max_proposals, flip_ratio,
            with_mask, with_crowd, with_label, with_semantic_seg, seg_prefix,
            seg_scale_factor, extra_aug, resize_keep_ratio, corruption,
            corruption_severity, skip_img_without_anno, test_mode)
        self.fill_instance = fill_instance
        self.center_fill_before = center_fill_before
        if self.fill_instance:
            print(
                'If there exist more than one parts in a mask, fill it first')
            if self.center_fill_before:
                print(
                    'The center is the average of coordinates of several parts'
                )
            else:
                print('after filling, calculate the center')

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
        img_meta = dict(ori_shape=ori_shape,
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

        _labels, _bbox_targets, _mask_targets = self.polar_target_single(
            gt_bboxes, gt_masks, gt_labels, concat_points,
            concat_regress_ranges)

        data['_gt_labels'] = DC(_labels)
        data['_gt_bboxes'] = DC(_bbox_targets)
        data['_gt_masks'] = DC(_mask_targets)
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
        return points.float()

    def polar_target_single(self, gt_bboxes, gt_masks, gt_labels, points,
                            regress_ranges):
        '''
        对feature map上的每个点确定用极坐标表示的bbox和轮廓点

        input:
        gt_bboxes: 表示图中所有实例的bbox shape (k,4)
        gt_masks: shape (k,768,1280)
        gt_labels: shape (k)
        points: 所有特征图上所有的点(对应到原图的位置)
        regress_ranges: 所有特征图上所有点（对应回原图的坐标）的回归值的范围
        output:
        labels: 每个点对应回原图的未知的类别（0为背景）
        bbox_taegets: 每个像素对应回原图的点距离其bbox的四条边的距离（bbox是按照mask的assign方法找的）
        mask_targets: 每个像素对应回原图的点如果在某bbox内或者在
        center_fill_before:在进行实例填充的情况下，若为True在用填充前各实例的中心的均值作为整个mask的中心；若为False，则用填充后的实例的中心作为质心
        '''
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] -
                                                           gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)  # [None]升维操作
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)  # 将原有的回归范围限制又复制了实例个数次
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        # xs ys 分别是points的x y坐标
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        # feature map上所有点对于gtbox的上下左右距离 [num_pix, num_gt, 4]
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # mask targets 也按照这种写 同时labels 得从bbox中心修改成mask重心
        mask_centers = []
        mask_contours = []

        # 第一步 先算k个重心  return [num_gt, 2]
        for i in range(gt_masks.shape[0]):
            mask = gt_masks[i]
            # add by amd for fill instance
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
            if len(contours) > 1 and self.fill_instance:
                # 对同一mask中的隔断实例进行填充
                if self.center_fill_before:
                    # 以每个隔断部分的中心的均值为质心
                    mask, cnt = self.fillInstance(mask)
                    _, contour = self.get_single_centerpoint(mask)
                else:
                    # 以填充后的mask的质心为中心
                    # cv2.imwrite('./vis/mask_old_{}.jpg'.format(i), mask * 255)
                    mask = self.fillInstance(mask)
                    cnt, contour = self.get_single_centerpoint(mask)
                    # cv2.imwrite('./vis/mask_{}.jpg'.format(i), mask * 255)
            else:
                cnt, contour = self.get_single_centerpoint(mask)

            # cnt, contour = self.get_single_centerpoint(mask)
            contour = contour[0]  # 面积最大的轮廓
            contour = torch.Tensor(contour).float()
            y, x = cnt

            mask_centers.append([x, y])
            mask_contours.append(contour)
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
        pos_inds = labels.nonzero().reshape(-1)  # 正样本下标

        mask_targets = torch.zeros(num_points, 36).float()

        pos_mask_ids = min_area_inds[pos_inds]
        for p, id in zip(pos_inds, pos_mask_ids):
            x, y = points[p]
            pos_mask_contour = mask_contours[id]

            dists, coords = self.get_36_coordinates(x, y, pos_mask_contour)
            mask_targets[p] = dists

        return labels, bbox_targets, mask_targets

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
        (gt_xs, gt_ys):图像中点的横纵坐标，需要判断点是否是中心像素
        '''
        center_y = mask_center[..., 0]
        center_x = mask_center[..., 1]
        center_gt = gt_bb.new_zeros(gt_bb.shape)
        # no gt,该图像不含实例
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
        求质心，公式等价于所有点在x，y轴上的均值
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

    def get_single_centerpoint(self, mask):
        '''
        input: 掩码
        output: 中心坐标、掩码中的所有轮廓(按照面积大小排序)
        '''
        contour, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_NONE)
        contour = list(contour)
        contour.sort(key=lambda x: cv2.contourArea(x),
                     reverse=True)  # only save the biggest one
        '''debug IndexError: list index out of range'''
        count = contour[0][:, 0, :]
        try:
            center = self.get_centerpoint(count)
        except:
            x, y = count.mean(axis=0)
            center = [int(x), int(y)]

        # max_points = 360
        # if len(contour[0]) > max_points:
        #     compress_rate = len(contour[0]) // max_points
        #     contour[0] = contour[0][::compress_rate, ...]
        return center, contour

    def get_36_coordinates(self, c_x, c_y, pos_mask_contour):
        '''
        input: 极心坐标, 掩码轮廓
        output: 36个点的极坐标表示, 36个点的极径
        找每个角度周围正负3的点, 如果他们在轮廓点中出现过, 则将他们到轮廓点的距离作为极径
        '''
        ct = pos_mask_contour[:, 0, :]
        x = ct[:, 0] - c_x
        y = ct[:, 1] - c_y
        # angle = np.arctan2(x, y)*180/np.pi
        angle = torch.atan2(x, y) * 180 / np.pi  # 点的位置就能反应角度,计算极角则可得到
        angle[angle < 0] += 360
        angle = angle.int()
        # dist = np.sqrt(x ** 2 + y ** 2)
        dist = torch.sqrt(x**2 + y**2)
        angle, idx = torch.sort(angle)
        dist = dist[idx]

        # 生成36个角度
        new_coordinate = {}
        for i in range(0, 360, 10):
            if i in angle:
                d = dist[angle == i].max()
                new_coordinate[i] = d
            elif i + 1 in angle:
                d = dist[angle == i + 1].max()
                new_coordinate[i] = d
            elif i - 1 in angle:
                d = dist[angle == i - 1].max()
                new_coordinate[i] = d
            elif i + 2 in angle:
                d = dist[angle == i + 2].max()
                new_coordinate[i] = d
            elif i - 2 in angle:
                d = dist[angle == i - 2].max()
                new_coordinate[i] = d
            elif i + 3 in angle:
                d = dist[angle == i + 3].max()
                new_coordinate[i] = d
            elif i - 3 in angle:
                d = dist[angle == i - 3].max()
                new_coordinate[i] = d

        distances = torch.zeros(36)

        for a in range(0, 360, 10):
            if not a in new_coordinate.keys():
                new_coordinate[a] = torch.tensor(1e-6)
                distances[a // 10] = 1e-6
            else:
                distances[a // 10] = new_coordinate[a]
        # for idx in range(36):
        #     dist = new_coordinate[idx * 10]
        #     distances[idx] = dist

        return distances, new_coordinate

    def fillInstance(self, instance):
        '''
        存在的多个实例/轮廓，找出所有轮廓点的外接框，然后取轮廓点所在的每行每列中离外接框最近的点（每个group中的部分轮廓点不会留下）。
        存在5个及以上的轮廓时，找轮廓点的先后连接顺序。（旅行商问题，实现是随机游走找的）
        各个group中轮廓点的顺序是按照外接框的顺时针顺序确定的
        '''
        if len(instance.shape) > 2:
            instance = instance[:, :, 0]
        instance = instance.astype(np.uint8)
        # _, contours, _ = cv.findContours(
        #     instance, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contours, _ = cv2.findContours(instance, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)

        if len(contours) > 1:
            # whole instance
            # result in shape (N,1,2)

            # computing areas
            edgePoints = contours[0]
            for i in range(1, len(contours)):
                edgePoints = np.concatenate((edgePoints, contours[i]),
                                            axis=0)  # 所有轮廓点，尺寸（N，1，2）

            dictEdgePoint = {}  # for later grouping
            for i in range(len(contours)):
                for j in range(contours[i].shape[0]):
                    e_x = str(contours[i][j][0][0])
                    e_y = str(contours[i][j][0][1])
                    dictEdgePoint[e_x + "_" +
                                  e_y] = [i, j]  # 像素(e_x,e_y)是第i条轮廓的第j个轮廓点

            # bbox of whole instance，得到所有实例不的最小外接框（不旋转）
            x, y, w, h = cv2.boundingRect(edgePoints)  # 外接框的左上点坐标和长宽

            # extract outline contour
            distanceMapUp = np.zeros((w + 1, 1))
            distanceMapUp.fill(np.inf)
            distanceMapDown = np.zeros((w + 1, 1))
            distanceMapDown.fill(-np.inf)
            distanceMapLeft = np.zeros((h + 1, 1))
            distanceMapLeft.fill(np.inf)
            distanceMapRight = np.zeros((h + 1, 1))
            distanceMapRight.fill(-np.inf)

            # 找到每一行、每一列最靠近外接框的点,但有的行、有的列不一定有点，则对应distance中的值为inf或-inf
            # 靠近左边线的点到顶点的x距离尽可能小，靠近右边线的点到顶点的x距离尽可能大
            for edgePoint in edgePoints:
                p_x = edgePoint[0][0]
                p_y = edgePoint[0][1]
                index_x = p_x - x
                index_y = p_y - y
                if index_y < distanceMapUp[index_x]:
                    distanceMapUp[index_x] = index_y
                if index_y > distanceMapDown[index_x]:
                    distanceMapDown[index_x] = index_y
                if index_x < distanceMapLeft[index_y]:
                    distanceMapLeft[index_y] = index_x
                if index_x > distanceMapRight[index_y]:
                    distanceMapRight[index_y] = index_x

            # grouping outline to original contours, it can make undirected points partially directed
            selected_points = []
            selected_info = {}  # 像素(e_x,e_y)是第i条轮廓的第j个轮廓点
            # 将每行每列最靠近边框的点按照顺时针顺序依次append
            for i in range(w + 1):  # 遍历边界框所有行和列，其中仅部分行列有像素因此需要判断inf，-inf
                if distanceMapUp[i] < np.inf:
                    e_x = int(i + x)
                    e_y = int(distanceMapUp[i] + y)
                    selected_points.append([e_x, e_y])
                    selected_info[str(e_x) + "_" +
                                  str(e_y)] = dictEdgePoint[str(e_x) + "_" +
                                                            str(e_y)]
            for i in range(h + 1):
                if distanceMapRight[i] > -np.inf:
                    e_x = int(distanceMapRight[i] + x)
                    e_y = int(i + y)
                    selected_points.append([e_x, e_y])
                    selected_info[str(e_x) + "_" +
                                  str(e_y)] = dictEdgePoint[str(e_x) + "_" +
                                                            str(e_y)]
            for i in range(w, -1, -1):
                if distanceMapDown[i] > -np.inf:
                    e_x = int(i + x)
                    e_y = int(distanceMapDown[i] + y)
                    selected_points.append([e_x, e_y])
                    selected_info[str(e_x) + "_" +
                                  str(e_y)] = dictEdgePoint[str(e_x) + "_" +
                                                            str(e_y)]
            for i in range(h, -1, -1):
                if distanceMapLeft[i] < np.inf:
                    e_x = int(distanceMapLeft[i] + x)
                    e_y = int(i + y)
                    selected_points.append([e_x, e_y])
                    selected_info[str(e_x) + "_" +
                                  str(e_y)] = dictEdgePoint[str(e_x) + "_" +
                                                            str(e_y)]

            selected_info = sorted(selected_info.items(),
                                   key=lambda x:
                                   (x[1], x[0]))  # 先按照轮廓序号、轮廓点序号排序

            groups = {}  # 记录对应轮廓上的点
            for item in selected_info:
                name = item[0]
                coord_x = name.split("_")[0]
                coord_y = name.split("_")[1]
                c = item[1][0]  # 轮廓序号
                try:
                    groups[c].append((int(coord_x), int(coord_y)))
                except KeyError:
                    groups[c] = [(int(coord_x), int(coord_y))]

            # connect group
            start_list = []
            end_list = []
            point_number_list = []
            for key in groups.keys():
                # inside each group, shift the array, so that the first and last point have biggest distance
                tempGroup = groups[key].copy()
                tempGroup.append(tempGroup.pop(0))  # 将起始点添加到末尾点之后
                distGroup = np.diag(
                    distance.cdist(groups[key], tempGroup,
                                   'euclidean'))  # 计算（0, 1） (1, 2)...(n,0)的距离
                max_index = np.argmax(distGroup)
                # 如果当前的首尾两点不是距离最远的点，则按照距离最远的点调整数组。groups中每个键存的点都是首尾最远
                if max_index != len(groups[key]) - 1:
                    groups[key] = groups[key][max_index+1:] + \
                        groups[key][:max_index+1]
                point_number_list.append(len(groups[key]))
                start_list.append(groups[key][0])
                end_list.append(groups[key][-1])

            # get center point here,中心是计算的多个轮廓的中心
            point_count = 0
            center_x = 0
            center_y = 0
            for i in range(len(start_list)):
                center_x += start_list[i][0]
                center_x += end_list[i][0]
                center_y += start_list[i][1]
                center_y += end_list[i][1]
                point_count += 2
            center_x /= point_count
            center_y /= point_count

            # calculate the degree based on center point
            degStartList = []
            for i in range(len(start_list)):
                deg = - \
                    np.arctan2(
                        1, 0) + np.arctan2(start_list[i][0]-center_x, start_list[i][1]-center_y)
                deg = deg * 180 / np.pi
                if deg < 0:
                    deg += 360
                degStartList.append(deg)

            # first solely consider the degree, construct a base solution
            best_path = np.argsort(degStartList)  # 按照起始点和中心连线的角度连接各个group
            best_path = np.append(best_path, best_path[0])

            # then consider distance, model it as asymmetric travelling salesman problem
            # note: add this step the solution is not necessarily better
            # note: if an object is relatively simple, i.e. <=3 area, do not need this
            # TODO: find a more robust solution here
            # 根据初始路径进行随机变换，找路径长度之和最大的路径
            if len(groups.keys()) > 4:
                distMatrix = distance.cdist(end_list, start_list, 'euclidean')

                MAX_ITER = 100
                count = 0
                while count < MAX_ITER:
                    path = best_path.copy()
                    start = np.random.randint(1,
                                              len(path) -
                                              1)  # start始终不为0，第一个点保持不变
                    if np.random.random() > 0.5:
                        while start - 2 <= 1:  # 保证start始终大于3
                            start = np.random.randint(1,
                                                      len(path) -
                                                      1)  # start始终不为0，第一个点保持不变
                        end = np.random.randint(1, start - 2)

                        path[end:start + 1] = path[end:start + 1][::-1]
                    else:
                        while start + 2 >= len(path) - 1:
                            start = np.random.randint(1, len(path) - 1)
                        end = np.random.randint(start + 2, len(path) - 1)

                        path[start:end + 1] = path[start:end + 1][::-1]
                    if self.compare_path(best_path, path, distMatrix):
                        count = 0
                        best_path = path
                    else:
                        count += 1
            final_points = []
            groupList = list(groups.keys())

            for i in range(len(best_path) - 1):
                # 按照best_path的顺序将group中对应轮廓的点加起来
                final_points += groups[groupList[best_path[i]]]
            final_points = np.array(final_points)

            # fill the break piece
            instance_id = instance.max()
            cv2.fillPoly(instance, [final_points],
                         (int(instance_id), 0, 0))  # (768,1280)
        if self.center_fill_before:
            return instance, (center_x, center_y)
        else:
            return instance

    def compare_path(self, path_1, path_2, distMatrix):
        sum1 = 0
        for i in range(1, len(path_1)):
            sum1 += distMatrix[path_1[i - 1]][path_1[i]]

        sum2 = 0
        for i in range(1, len(path_2)):
            sum2 += distMatrix[path_2[i - 1]][path_2[i]]

        return sum1 > sum2

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)

            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
