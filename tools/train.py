# coding: utf-8

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
from mmcv.runner import load_checkpoint

import torch.nn as nn
from mmcv.cnn import constant_init, normal_init
from mmdet.models.utils import Scale

__VERSION__ = '1.0.rc0+unknown'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from',
                        help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument('--gpus',
                        type=int,
                        default=1,
                        help='number of gpus to use '
                        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale-lr',
                        action='store_true',
                        help='automatically scale lr with the number of gpus')

    parser.add_argument('--debug',
                        action='store_true',
                        help='use main or test')  # 只要对变量进行传参，则值为True
    parser.add_argument('--mode', type=int, default=1, help='tmp use')
    parser.add_argument('--train_refine',
                        action='store_true',
                        help='only train refine module')  # 只要对变量进行传参，则值为True

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(cfg.model,
                           train_cfg=cfg.train_cfg,
                           test_cfg=cfg.test_cfg)

    if args.train_refine:
        # 加载预训练init模型
        checkpoint = load_checkpoint(
            model,
            '/apdcephfs/private_amandaaluo/PolarMask/work_dirs/polarmask_refine_polar_both/latest.pth',
            map_location='cpu')

        # 固定参数
        for a in model.parameters():
            a.requires_grad = False
        trainable_modules = [
            model.bbox_head.refine, model.bbox_head.scale_refine
        ]
        for module in trainable_modules:
            for a in module.parameters():
                a.requires_grad = True

        # 对可训练模块进行初始化
        for module in trainable_modules:
            for m in module.modules():
                # 在遍历时会出现更高层级的对象，因此需要判断是否属于某实例
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
                elif isinstance(m, nn.GroupNorm):
                    constant_init(m, 1)
                elif isinstance(m, Scale):
                    m.scale = nn.Parameter(torch.tensor(1.0,
                                                        dtype=torch.float))

        # print(model.bbox_head.refine.weight.mean())
        # print([a for a in model.bbox_head.refine.parameters()])
        # print([a for a in model.bbox_head.scale_refine.parameters()])

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data

        # cfg.checkpoint_config.meta = dict(mmdet_version=__version__,
        #                                   config=cfg.text,
        #                                   CLASSES=datasets[0].CLASSES)
        cfg.checkpoint_config.meta = dict(mmdet_version=__VERSION__,
                                          config=cfg.text,
                                          CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(model,
                   datasets,
                   cfg,
                   distributed=distributed,
                   validate=args.validate,
                   logger=logger)


def test():
    from tqdm import trange
    import cv2
    print('debug mode ' * 10)
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = 1

    dataset = build_dataset(cfg.data.train)
    embed(header='123123')

    def visual(i):
        img = dataset[i]['img'].data
        img = img.permute(1, 2, 0) + 100
        img = img.data.cpu().numpy()
        cv2.imwrite('./trash/resize_v1.jpg', img)

    # embed(header='check data resizer')


def capacity_test():
    print(1)
    print(2)
    print(3)
    print(4)


if __name__ == '__main__':
    args = parse_args()
    if not args.debug:
        main()
        # from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp_wrapper = lp(capacity_test)
        # lp.print_stats()
    else:
        test()
