from __future__ import division
import argparse
import os

import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

from IPython import embed

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')

    parser.add_argument(
        '--debug',
        action='store_true',
        help='use main or test')  # 只要对变量进行传参，则值为True
    parser.add_argument(
        '--mode',
        type=int,
        default=1,
        help='tmp use')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def test():
    from tqdm import trange
    print('debug mode '*10)
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = 1

    dataset = build_dataset(cfg.data.train)
    embed(header='123123')


def visual_img(dataset, i):
    img = dataset[i]['img'].data
    img = img.permute(1, 2, 0) + 100
    img = img.data.cpu().numpy()
    vis_path = './vis/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    cv2.imwrite(os.path.join(vis_path, 'img'+str(i)+'.jpg'), img)


def visual_mask(dataset, i):
    masks = dataset[i]['gt_masks'].data
    for j in range(masks.shape[0]):
        mask = masks[j, :, :]*255
        img_r = mask[:, :, np.newaxis]
        img = np.concatenate([img_r, img_r, img_r], axis=2)
        vis_path = './vis/'
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        cv2.imwrite(os.path.join(vis_path, 'img' +
                                 str(i)+'_mask{}.jpg'.format(j)), img)


def upper_bound_analysis():
    '''
    dataset数据结构
    '''
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = 1

    dataset = build_dataset(cfg.data.train)
    # for i in range(5):
    #     visual_img(dataset, i)
    #     visual_mask(dataset, i)
    # for i in range(5):
    masks = dataset[0]['ft_masks'].data
    for j in range(masks.shape[0]):
        mask = np.float(masks[j, :, :] * 255)
        Harris_detector = cv2.cornerHarris(gray_img, 2, 3, 0.04)


    # 遍历训练集/验证集，计算以10度等间隔取的点为gt，和不受角度约束取的点（36个..）
    # 对每个实例取mask
    # 计算mask的角点


if __name__ == '__main__':
    args = parse_args()
    upper_bound_analysis()
