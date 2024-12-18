# Visualize rusluts
# for basline, only show result
# for improve, show result, center, radius, angle

from mmdet.apis import init_detector, inference_detector, show_result
import pycocotools.mask as maskUtils
import mmcv
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
from mmcv import Config
from mmdet.datasets import build_dataset

colors = [(0, 255, 0)]
for m in range(100):
    colors.append(
        tuple(np.random.randint(0, 256, (1, 3), dtype=np.uint8).tolist()[0]))


def colorMask(img, segm):
    mask = maskUtils.decode(segm).astype(np.bool)
    color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
    img[mask] = img[mask] * 0.3 + color_mask * 0.7
    return img


# 只适用于前向输出
def show_result_compare(img_name,
                        baseline_result,
                        result,
                        class_names,
                        output_path,
                        gt_result=None,
                        score_thr=0.2,
                        fig_size=(15, 10),
                        original=False,
                        vis_seperate=False,
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
        original : 可视化的是否为polarmask的原论文结果
        vis_seperate : 使用分开的图像处理或是将baseline和实验结果放在同一张图
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if original:
        baseline_result = show_result_xy(IMG_PATH + img_name,
                                         baseline_result,
                                         class_names,
                                         score_thr=score_thr,
                                         show=False,
                                         color_mask=color_mask,
                                         draw_contour_line=draw_contour_line,
                                         draw_contour_point=draw_contour_point,
                                         draw_center=draw_center,
                                         draw_bbox=draw_bbox)
    else:
        # 适用于coarse-to-fine中的仅用init作为baseline的情况
        baseline_result = show_result_xy(IMG_PATH + img_name,
                                         baseline_result,
                                         class_names,
                                         score_thr=score_thr,
                                         show=False,
                                         color_mask=color_mask,
                                         draw_contour_line=draw_contour_line,
                                         draw_contour_point=draw_contour_point,
                                         draw_center=draw_center,
                                         draw_bbox=draw_bbox)

    result = show_result_xy(IMG_PATH + img_name,
                            result,
                            class_names,
                            score_thr=0.2,
                            show=False,
                            color_mask=color_mask,
                            draw_contour_line=draw_contour_line,
                            draw_contour_point=draw_contour_point,
                            draw_center=draw_center,
                            draw_bbox=draw_bbox)

    if vis_seperate:
        # # visualize seperately
        # cv2.imwrite(output_path + 'init_' + img_name, baseline_result)
        cv2.imwrite(output_path + 'refine_' + img_name, result)
        # cv2.imwrite(output_path + 'gt_' + img_name, gt_result)
    else:
        if gt_result is None:
            plt.figure(figsize=(25, 15))
            plt.subplot(121)
            plt.imshow(mmcv.bgr2rgb(baseline_result))
            plt.subplot(122)
            plt.imshow(mmcv.bgr2rgb(result))
            plt.savefig(output_path + 'compare_{}'.format(img_name))
        else:
            plt.figure(figsize=(25, 15))
            plt.subplot(131)
            plt.axis('off')
            plt.imshow(mmcv.bgr2rgb(baseline_result))
            plt.subplot(132)
            plt.axis('off')
            plt.imshow(mmcv.bgr2rgb(result))
            plt.subplot(133)
            plt.axis('off')
            plt.imshow(mmcv.bgr2rgb(gt_result))
            plt.axis('off')
            plt.savefig(output_path + 'compare_{}'.format(img_name))


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

        # self-processing
        # if img_name.split('/')[-1] == '000000361571.jpg':
        #     if len(inds) > 1:
        #         inds = [inds[1]]
        # if img_name.split('/')[-1] == '000000236845.jpg':
        #     if len(inds) == 2:
        #         inds = [inds[1], inds[0]]
        #     if len(inds) == 3:
        #         inds = [inds[2], inds[0], inds[1]]
        # if img_name.split('/')[-1] == '000000111036.jpg':
        #     inds = [inds[1]]
        # if img_name.split('/')[-1] == '000000147223.jpg':
        #     if len(inds) > 4:
        #         inds = inds[:7]
        #         # inds = inds[1:6] + [inds[0]]

        for k in range(len(inds)):
            mask = maskUtils.decode(segms[inds[k]]).astype(np.uint8)
            cv2.imwrite('./vis/panNuke/inst_map/'+img_name.split('/')[-1].split('.png')[0]+'_'+str(inds[k])+'.png', mask*255)


        for k in range(len(inds)):
            i = inds[k]
            segm = segms[i]

            # # for init images
            # if img_name.split('/')[-1] == '0062.png':
            #     if i in [0,11,12]:
            #         continue
            # if img_name.split('/')[-1] == '0685.png':
            #     if i in [22,23]:
            #         continue
            # if img_name.split('/')[-1] == '1370.png':
            #     if i in [32,58]:
            #         continue
            
            if img_name.split('/')[-1] == '0062.png':
                if i in [0,20,27,37,45]:
                    continue
            if img_name.split('/')[-1] == '0208.png':
                if i in [44,53]:
                    continue
            if img_name.split('/')[-1] == '0438.png':
                if i in [28]:
                    continue
            if img_name.split('/')[-1] == '0685.png':
                if i in [14,12,10,83,2,3]:
                    continue
            if img_name.split('/')[-1] == '1370.png':
                if i in [7, 46]:
                    continue

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
                             (x, y), colors[k], 2)

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



def show(img_names,
         config_file,
         checkpoint_file,
         output_path,
         compare=True,
         baseline_config_file=None,
         baseline_checkpoint_file=None,
         color_mask=True,
         draw_contour_line=False,
         draw_contour_point=False,
         draw_center=False,
         draw_bbox=False,
         compare_w_gt=True):
    '''
    compare:可视化是单图展示或两图对比
    '''
    original = False
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    if compare:
        if baseline_config_file is None and baseline_checkpoint_file is None:
            baseline_config_file = '/data/lf/projects/polar_baseline/configs/polarmask/4gpu/polar_768_1x_r50.py'
            baseline_checkpoint_file = '/data/lf/projects/polar_baseline/work_dirs/panNuke/polar_768_1x_r50_3x/latest.pth'
            original = True
        elif baseline_config_file == '/data/lf/projects/polar_baseline/configs/polarmask/4gpu/polar_768_1x_r50.py' and baseline_checkpoint_file == '/data/lf/projects/polar_baseline/work_dirs/panNuke/polar_768_1x_r50_3x/latest.pth':
            original = True

        baseline_model = init_detector(baseline_config_file,
                                       baseline_checkpoint_file,
                                       device='cuda:0')

    # if isinstance(img_names, list):
    #     for i in range(len(dataset)):
    #         # img_name = img_names[i]
    #         if not dataset[i]['img_meta'].data['file_name'] in img_names:
    #             continue
    #         else:
                
    #             if compare:
    #                 if compare_w_gt:
    #                     # redefine img name
    #                     img_name = dataset[i]['img_meta'].data['file_name']
    #                     baseline_result = inference_detector(baseline_model,
    #                                                         IMG_PATH + img_name,
    #                                                         show=True)
    #                     result = inference_detector(model,
    #                                                 IMG_PATH + img_name,
    #                                                 show=True)
    #                     gt_result = draw_gt(dataset[i])

    #                     show_result_compare(img_name,
    #                                         baseline_result,
    #                                         result,
    #                                         model.CLASSES,
    #                                         output_path,
    #                                         gt_result=gt_result,
    #                                         original=original,
    #                                         color_mask=color_mask,
    #                                         draw_contour_line=draw_contour_line,
    #                                         draw_contour_point=draw_contour_point,
    #                                         draw_center=draw_center,
    #                                         draw_bbox=draw_bbox,
    #                                         vis_seperate=True)
    #                 else:
    #                     baseline_result = inference_detector(baseline_model,
    #                                                         IMG_PATH + img_name,
    #                                                         show=True)
    #                     result = inference_detector(model,
    #                                                 IMG_PATH + img_name,
    #                                                 show=True)
    #                     show_result_compare(img_name,
    #                                         baseline_result,
    #                                         result,
    #                                         model.CLASSES,
    #                                         output_path,
    #                                         original=original,
    #                                         color_mask=color_mask,
    #                                         draw_contour_line=draw_contour_line,
    #                                         draw_contour_point=draw_contour_point,
    #                                         draw_center=draw_center,
    #                                         draw_bbox=draw_bbox)
    #             else:
    #                 # 单图展示
    #                 result = inference_detector(model,
    #                                             IMG_PATH + img_name,
    #                                             show=True)
    #                 show_result_pyplot(img_name,
    #                                 result,
    #                                 model.CLASSES,
    #                                 output_path,
    #                                 color_mask=color_mask,
    #                                 draw_contour_line=draw_contour_line,
    #                                 draw_contour_point=draw_contour_point,
    #                                 draw_center=draw_center,
    #                                 draw_bbox=draw_bbox)

    if isinstance(img_names, list):
        for i in range(len(img_names)):
            img_name = img_names[i]
                
            if compare:
                if compare_w_gt:
                    # redefine img name
                    img_name = dataset[i]['img_meta'].data['file_name']
                    baseline_result = inference_detector(baseline_model,
                                                        IMG_PATH + img_name,
                                                        show=True)
                    result = inference_detector(model,
                                                IMG_PATH + img_name,
                                                show=True)
                    gt_result = draw_gt(dataset[i])

                    show_result_compare(img_name,
                                        baseline_result,
                                        result,
                                        model.CLASSES,
                                        output_path,
                                        gt_result=gt_result,
                                        original=original,
                                        color_mask=color_mask,
                                        draw_contour_line=draw_contour_line,
                                        draw_contour_point=draw_contour_point,
                                        draw_center=draw_center,
                                        draw_bbox=draw_bbox,
                                        vis_seperate=True)
                else:
                    baseline_result = inference_detector(baseline_model,
                                                        IMG_PATH + img_name,
                                                        show=True)
                    result = inference_detector(model,
                                                IMG_PATH + img_name,
                                                show=True)
                    show_result_compare(img_name,
                                        baseline_result,
                                        result,
                                        model.CLASSES,
                                        output_path,
                                        original=original,
                                        color_mask=color_mask,
                                        draw_contour_line=draw_contour_line,
                                        draw_contour_point=draw_contour_point,
                                        draw_center=draw_center,
                                        draw_bbox=draw_bbox,
                                        vis_seperate=True)
            else:
                # 单图展示
                result = inference_detector(model,
                                            IMG_PATH + img_name,
                                            show=True)
                show_result_pyplot(img_name,
                                result,
                                model.CLASSES,
                                output_path,
                                color_mask=color_mask,
                                draw_contour_line=draw_contour_line,
                                draw_contour_point=draw_contour_point,
                                draw_center=draw_center,
                                draw_bbox=draw_bbox)


def draw_gt(data):
    # img = data['img'].data
    # img_bgr = img.permute(1, 2, 0) 
    # img_bgr = img_bgr.data.cpu().numpy().astype(np.uint8)
    img_bgr = cv2.imread('/data/lf/dataset/panNuke/img/fold_2/'+data['img_meta'].data['file_name'])
    masks = data['gt_masks'].data # 尺寸：(5,H/8,W/8)，每一层采样点，绘制到(96,160)
    for i in range(len(masks)):
        mask = masks[i]
        res, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, res, -1,  colors[i], 2)


    return img_bgr
    
# images to visualize
IMG_PATH = '/data/lf/dataset/panNuke/img/fold_2/'
# file_names = os.listdir('/data/img_quality/')
# file_names = os.listdir(IMG_PATH)
# file_no = [62, 204, 208, 438, 685, 979, 1370]
file_no = [1370]
# file_names = [IMG_PATH+str(no).zfill(4)+'.png' for no in file_no]
file_names = [str(no).zfill(4)+'.png' for no in file_no]

cfg = Config.fromfile('./demo/polar_init_refine_r50_centerness_polar_heatmap_5_10_3x.py')
dataset = build_dataset(cfg.data.train)


# models to visulize

# original polarMask-res50
config_file1 = '/data/lf/projects/polar_baseline/configs/panNuke/polar_768_1x_r50.py'
checkpoint_file1 = '/data/lf/projects/polar_baseline/work_dirs/panNuke/polar_768_1x_r50_3x/latest.pth'

# init
config_file2 = './configs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10_3x_init.py'
checkpoint_file2 = './work_dirs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10_3x/latest.pth'

# # # 只含refine
# config_file = './configs/polarmask_refine/1gpu/polar_init_refine_r101_centerness_polar.py'
# checkpoint_file = './work_dirs/polar_init_refine_r101_centerness_polar/latest.pth'

# refine+hbb
config_file = './configs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10_3x.py'
checkpoint_file = './work_dirs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10_3x/latest.pth'

output_path = 'vis/panNuke/' + config_file.split('/')[-1].split('.')[0] + '/refine/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

show(file_names,
     config_file1,
     checkpoint_file1,
     output_path,
     compare=True,
     baseline_config_file=config_file2,
     baseline_checkpoint_file=checkpoint_file2,
     color_mask=False,
     draw_contour_line=True,
     draw_contour_point=False,
     draw_center=False,
     draw_bbox=False,
     compare_w_gt=False)
