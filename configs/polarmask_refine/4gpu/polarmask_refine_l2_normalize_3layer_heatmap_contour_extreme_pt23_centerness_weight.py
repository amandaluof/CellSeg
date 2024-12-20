# coding: utf-8

import math

# model settings
model = dict(
    type='PolarMask_Angle',
    pretrained='/apdcephfs/private_amandaaluo/resnet50_caffe-788b5fa3.pth',
    backbone=dict(type='ResNet',
                  depth=50,
                  num_stages=4,
                  out_indices=(0, 1, 2, 3),
                  frozen_stages=1,
                  norm_cfg=dict(type='BN', requires_grad=False),
                  style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(type='PolarMask_Refine_Head',
                   num_classes=81,
                   in_channels=256,
                   stacked_convs=4,
                   stacked_init_convs=4,
                   stacked_refine_convs=2,
                   stacked_additinal_cls_convs=4,
                   refine_mask=True,
                   feat_channels=256,
                   strides=[8, 16, 32, 64, 128],
                   normalize_factor=1 / math.sqrt(1280**2 + 768**2),
                   loss_cls=dict(type='FocalLoss',
                                 use_sigmoid=True,
                                 gamma=2.0,
                                 alpha=0.25,
                                 loss_weight=1.0),
                   loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                   loss_centerness=dict(type='CrossEntropyLoss',
                                        use_sigmoid=True,
                                        loss_weight=0.5),
                   loss_mask_init=dict(type='L2Loss',
                                   beta=1.0,
                                   loss_weight=0.3),
                   loss_mask_refine=dict(type='L2Loss',
                                   beta=1.0,
                                   loss_weight=0.6),
                   loss_heatmap=dict(type='BCEFocalLoss',
                                     gamma=2,
                                     alpha=0.25,
                                     reduction='sum',
                                     loss_weight=1),
                   loss_pts_bbox_init=dict(type='L2Loss', beta=1.0, loss_weight=2),
                   loss_pts_bbox_refine=dict(type='L2Loss', beta=1.0, loss_weight=3),
                   additional_cls_branch=True,
                   additional_cls_single=True,
                   all_instances_heatmap=True))

# training and testing settings
train_cfg = dict(assigner=dict(type='MaxIoUAssigner',
                               pos_iou_thr=0.5,
                               neg_iou_thr=0.4,
                               min_pos_iou=0,
                               ignore_iof_thr=-1),
                 allowed_border=-1,
                 pos_weight=-1,
                 debug=False)
test_cfg = dict(nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms', iou_thr=0.5),
                max_per_img=100)
# dataset settings
dataset_type = 'Coco_Seg_Dataset_gt'
data_root = '/apdcephfs/private_amandaaluo/coco/'
img_norm_cfg = dict(mean=[102.9801, 115.9465, 122.7717],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False)
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        img_scale=(1280, 768),
        img_norm_cfg=img_norm_cfg,
        # size_divisor=0,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=False,
        with_label=True,
        resize_keep_ratio=False,
        num_sample_points=36,
        explore_times=0,
        fixed_gt=True,
        sort=False,
        polar_coordinate=False,
        ensure_inner=False,
        extreme_point=True,
        normalize_factor=1 / math.sqrt(1280**2 + 768**2),
        sample_point_heatmap=True,
        heatmap_contour=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1280, 768),
        img_norm_cfg=img_norm_cfg,
        # size_divisor=0,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        resize_keep_ratio=False,
        num_sample_points=36,
        explore_times=0,
        fixed_gt=True,
        sort=False,
        polar_coordinate=False,
        ensure_inner=False),
    test=dict(type=dataset_type,
              ann_file=data_root + 'annotations/instances_val2017.json',
              img_prefix=data_root + 'val2017/',
              img_scale=(1280, 768),
              img_norm_cfg=img_norm_cfg,
              size_divisor=32,
              flip_ratio=0,
              with_mask=False,
              with_crowd=False,
              with_label=False,
              resize_keep_ratio=False,
              test_mode=True,
              num_sample_points=36,
              explore_times=0,
              fixed_gt=True,
              sort=False,
              polar_coordinate=False,
              extreme_point=True,
              normalize_factor=1 / math.sqrt(1280**2 + 768**2),
              ensure_inner=False,
              sample_point_heatmap=True,
              heatmap_contour=True))
# optimizer
lr_ratio = 1

optimizer = dict(type='SGD',
                 lr=0.01 * lr_ratio,
                 momentum=0.9,
                 weight_decay=0.0001,
                 paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1.0 / 3 / lr_ratio,
                 step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/trash'
load_from = None
resume_from = None
workflow = [('train', 1)]  # only training,no validate;
