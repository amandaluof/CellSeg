# coding: utf-8

from __future__ import division
import argparse
import os
import cv2
import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from IPython import embed
from mmcv.runner import load_checkpoint

import torch.nn as nn
from mmcv.cnn import constant_init, normal_init
from mmdet.models.utils import Scale

import numpy as np

__VERSION__ = '1.0.rc0+unknown'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()

    return args

def visual(dataset, i):
    img = dataset[i]['img'].data
    img = img.permute(1, 2, 0) 
    img = img.data.cpu().numpy()
    mask = dataset[i]['_gt_sample_heatmap'].data # 尺寸：(5,H/8,W/8)，每一层采样点，绘制到(96,160)
    mask  = torch.sum(mask, 0) 
    print(np.unique(mask))
    mask = mask.data.cpu().numpy() 
    mask = (mask * 255).astype(np.uint8)
    mask = np.concatenate([mask[:,:,None], mask[:,:,None], mask[:,:,None]], 2)
    cv2.imwrite('./imgs/img_{}.jpg'.format(i), img)
    cv2.imwrite('./imgs/mask_{}.jpg'.format(i), mask)
    blank = np.ones((512,20,3)) * 255
    mask = cv2.resize(mask, (512,512))
    output = np.concatenate([img, blank, mask], 1)
    cv2.imwrite('./imgs/output_{}.jpg'.format(i), output)


def test():
    from tqdm import trange
    import cv2
    print('debug mode ' * 10)
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = 1

    dataset = build_dataset(cfg.data.train)
    for i in range(10):
        visual(dataset, i)


if __name__ == '__main__':
    test()
