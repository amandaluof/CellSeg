# Visualize rusluts
# for basline, only show result
# for improve, show result, center, radius, angle

from mmdet.models.anchor_heads import polarmask_double_gt_head
from mmdet.apis import init_detector, inference_detector, show_result
import pycocotools.mask as maskUtils
import mmcv
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
import argparse

colors = [(0, 255, 0)]
for m in range(100):
    colors.append(
        tuple(np.random.randint(0, 256, (1, 3), dtype=np.uint8).tolist()[0]))


def parse_args():
    parser = argparse.ArgumentParser(description='Visulize')
    parser.add_argument(
        '--config',
        default=
        './configs/polarmask_refine/1gpu/polar_init_refine_r101_centerness_polar_heatmap_revise_4_10.py',
        help='train config file path')
    parser.add_argument(
        '--checkpoint',
        default=
        './work_dirs/polar_init_refine_r101_centerness_polar_heatmap_revise_4_10/latest.pth',
        help='checkpoint file path')
    parser.add_argument('--dataset',
                        default='val',
                        help=' dataset split of the test img')
    parser.add_argument('--img',
                        default=['000000236845.jpg'],
                        help=' img list to test')
    parser.add_argument('--color_mask',
                        action='store_true',
                        help='color masks of instances')  # 只要对变量进行传参，则值为True
    parser.add_argument(
        '--contour_line',
        action='store_true',
        help='draw contour lines of instances')  # 只要对变量进行传参，则值为True
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # images to visualize
    file_names = args.img
    #file_names = os.listdir('/data/img_quality/') # test结果

    # models to visulize
    # init
    # config_file = './configs/polarmask_refine/1gpu/polar_init_refine_r101_centerness_polar_init.py'
    # checkpoint_file = './work_dirs/polar_init_refine_r101_centerness_polar/latest.pth'

    # # 只含refine
    # config_file = './configs/polarmask_refine/1gpu/polar_init_refine_r101_centerness_polar.py'
    # checkpoint_file = './work_dirs/polar_init_refine_r101_centerness_polar/latest.pth'

    # refine+正则
    # config_file = './configs/polarmask_refine/1gpu/polar_init_refine_r101_centerness_polar_heatmap_revise_4_10.py'
    # checkpoint_file = './work_dirs/polar_init_refine_r101_centerness_polar_heatmap_revise_4_10/latest.pth'
    config_file = args.config
    checkpoint_file = args.checkpoint

    output_path = 'vis/' + config_file.split('/')[-1].split('.')[0] + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    show(file_names,
         config_file,
         checkpoint_file,
         output_path,
         color_mask=args.color_mask,
         draw_contour_line=args.contour_line,
         draw_contour_point=False,
         draw_center=False,
         draw_bbox=False)


def colorMask(img, segm):
    mask = maskUtils.decode(segm).astype(np.bool)
    color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
    img[mask] = img[mask] * 0.3 + color_mask * 0.7
    return img


# 可视化coarse-to-fine类型的模型的代码
def show_result_xy(img,
                   result,
                   class_names,
                   score_thr=0.3,
                   wait_time=0,
                   show=True,
                   out_file=None,
                   color_mask=True,
                   draw_contour_line=False,
                   draw_contour_point=False,
                   draw_center=False,
                   draw_bbox=False):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img_name = img
    img = mmcv.imread(img)
    img = img.copy()

    if isinstance(result, tuple):
        if len(result) == 3:
            bbox_result, segm_result, xy_result = result
        elif len(result) == 4:
            bbox_result, segm_result, xy_result, center_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    xy = np.vstack(xy_result)
    if len(result) == 4:
        centers = np.vstack(center_result)

    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for k in range(len(inds)):
            i = inds[k]
            segm = segms[i]
            # color the mask
            if color_mask:
                img = colorMask(img, segm)

            # draw the contour line
            if draw_contour_line:
                for j in range(36):
                    '''coordiante from get_bboxes() is deviated'''
                    x = int(xy[i, 0, j])
                    y = int(xy[i, 1, j])
                    cv2.line(img, (int(xy[i, 0, j - 1]), int(xy[i, 1, j - 1])),
                             (x, y), colors[k], 3)

            # draw contour points
            if draw_contour_point:
                for j in range(36):
                    '''coordiante from get_bboxes() is deviated'''
                    x = int(xy[i, 0, j])
                    y = int(xy[i, 1, j])
                    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

            # draw center
            if len(result) == 4 and draw_center:
                center_x = int(centers[i, 0])
                center_y = int(centers[i, 1])
                cv2.circle(img, (center_x, center_y), 2, (255, 0, 0))

            # draw bounding boxes
            if draw_bbox:
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = np.concatenate(labels)
                mmcv.imshow_det_bboxes(img,
                                       bboxes,
                                       labels,
                                       class_names=class_names,
                                       score_thr=score_thr,
                                       show=show,
                                       wait_time=wait_time,
                                       out_file=out_file)
    if not (show or out_file):
        return img


# 可视化单图结果
def show_result_pyplot(img_name,
                       result,
                       class_names,
                       output_path,
                       score_thr=0.3,
                       fig_size=(15, 10),
                       color_mask=True,
                       draw_contour_line=False,
                       draw_contour_point=False,
                       draw_center=False,
                       draw_bbox=False):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # basline
    if len(result) == 2:
        img = show_result(IMG_PATH + img_name,
                          result,
                          class_names,
                          score_thr=score_thr,
                          show=False)
        plt.figure(figsize=fig_size)
        plt.imshow(mmcv.bgr2rgb(img))
        plt.savefig(output_path + img_name)

    # init/refine
    elif len(result) == 4:
        img = show_result_xy(IMG_PATH + img_name,
                             result,
                             class_names,
                             score_thr=score_thr,
                             show=False,
                             color_mask=color_mask,
                             draw_contour_line=draw_contour_line,
                             draw_contour_point=draw_contour_point,
                             draw_center=draw_center,
                             draw_bbox=draw_bbox)
        # plt.figure(figsize=fig_size)
        # plt.imshow(mmcv.bgr2rgb(img))
        # plt.savefig(output_path + img_name)
        cv2.imwrite(output_path + img_name, img)

    # polar_angle
    elif len(result) == 5:
        img = show_result_pro(IMG_PATH + img_name,
                              result,
                              class_names,
                              score_thr=score_thr,
                              show=False)
        plt.figure(figsize=fig_size)
        plt.imshow(mmcv.bgr2rgb(img))
        plt.savefig(output_path + img_name)


# 可视化预(角度，极径)的方案
def show_result_pro(img_name,
                    result,
                    class_names,
                    score_thr=0.3,
                    wait_time=0,
                    show=True,
                    out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img_name)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result, center_result, radius_result, angle_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    center_results = np.vstack(center_result)
    radius_results = np.vstack(radius_result)
    angle_results = np.vstack(angle_result)

    # find the center when exapand the size of test image to (768,1280)
    tmp_centers = center_results.copy()
    tmp_centers[:, 0] = tmp_centers[:, 0] * (1280 / img.shape[1])
    tmp_centers[:, 1] = tmp_centers[:, 1] * (768 / img.shape[0])

    # draw segmentation masks
    record = {}
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
            cv2.circle(img,
                       (int(center_results[i, 0]), int(center_results[i, 1])),
                       5, (0, 0, 255))

            # For visualize the result of polarmask_double_gt_head
            from mmdet.models.anchor_heads.polarmask_double_gt_head import distance2mask
            import torch

            coordinate = distance2mask(
                torch.Tensor(tmp_centers[i][None, :]).cuda(),
                torch.Tensor(radius_results[i]).cuda(),
                torch.Tensor(angle_results[i]).cuda()).cpu().numpy()[0, :, :].T

            for j in range(len(coordinate)):
                '''coordiante from get_bboxes() is deviated'''
                x = int(coordinate[j, 0] / (1280 / img.shape[1]))
                y = int(coordinate[j, 1] / (768 / img.shape[0]))

                print(x, y)
                cv2.circle(img, (x, y), 5, (0, 0, 255))
            print('radius', radius_results[i])
            print('angle', angle_results[i] / math.pi * 180)
            print(((angle_results[i] / math.pi * 180) > 1).sum())
            record['radius_{}'.format(i)] = radius_results[i]
            record['angle_{}'.format(i)] = angle_results[i] / math.pi * 180

    import pandas as pd
    df = pd.DataFrame(record)
    res_name = img_name.split('/')[-1].split('.')[0]
    df.to_csv('./vis/train/' + '{}.csv'.format(res_name))

    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(img,
                           bboxes,
                           labels,
                           class_names=class_names,
                           score_thr=score_thr,
                           show=show,
                           wait_time=wait_time,
                           out_file=out_file)
    if not (show or out_file):
        return img


def show(img_names,
         config_file,
         checkpoint_file,
         output_path,
         color_mask=True,
         draw_contour_line=False,
         draw_contour_point=False,
         draw_center=False,
         draw_bbox=False):
    original = False
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    if isinstance(img_names, list):
        for img_name in img_names:
            # 单图展示
            result = inference_detector(model, IMG_PATH + img_name, show=True)
            show_result_pyplot(img_name,
                               result,
                               model.CLASSES,
                               output_path,
                               color_mask=color_mask,
                               draw_contour_line=draw_contour_line,
                               draw_contour_point=draw_contour_point,
                               draw_center=draw_center,
                               draw_bbox=draw_bbox)


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'val':
        IMG_PATH = '/youtu/xlab-team2/platform_data/amanda/coco/val2017/'
    elif args.dataset == 'test':
        IMG_PATH = '/youtu/xlab-team2/platform_data/amanda/coco/test2017/'
    main()
