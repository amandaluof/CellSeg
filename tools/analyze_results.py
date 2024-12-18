import argparse
import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as mask_util
# from mmcv import Config, DictAction
from mmcv import Config

from mmdet.core.evaluation import eval_map
# from mmdet.core.visualization import imshow_gt_det_bboxes
# from mmdet.datasets import build_dataset, get_loading_pipeline
from mmdet.datasets import build_dataset
import os
import cv2
import random


def bbox_map_eval(det_result, annotation):
    """Evaluate mAP of single image det result.

    Args:
        det_result (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotation (dict): Ground truth annotations where keys of
             annotations are:

            - bboxes: numpy array of shape (n, 4)
            - labels: numpy array of shape (n, )
            - bboxes_ignore (optional): numpy array of shape (k, 4)
            - labels_ignore (optional): numpy array of shape (k, )

    Returns:
        float: mAP
    """

    # use only bbox det result
    if isinstance(det_result, tuple):
        bbox_det_result = [det_result[0]]
    else:
        bbox_det_result = [det_result]

    # mAP
    iou_thrs = np.linspace(.5, 0.95, int(
        np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    mean_aps = []

    for thr in iou_thrs:
        mean_ap, _ = eval_map(
            bbox_det_result, [annotation], iou_thr=thr, logger='silent')
        mean_aps.append(mean_ap)

    return sum(mean_aps) / len(mean_aps)


def bbox_map_eval_total(det_results, annotations):
    """Evaluate mAP of single image det result.

    Args:
        det_result (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotation (dict): Ground truth annotations where keys of
             annotations are:

            - bboxes: numpy array of shape (n, 4)
            - labels: numpy array of shape (n, )
            - bboxes_ignore (optional): numpy array of shape (k, 4)
            - labels_ignore (optional): numpy array of shape (k, )

    Returns:
        float: mAP
    """

    # use only bbox det result
    if isinstance(det_results[0], tuple):
        bbox_det_results = [det_result[0] for det_result in det_results]
    else:
        bbox_det_results = det_results

    # mAP
    iou_thrs = np.linspace(.5, 0.95, int(
        np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    mean_aps = []

    for thr in iou_thrs:
        mean_ap, _ = eval_map(bbox_det_results, annotations,
                              iou_thr=thr, logger='silent')
        mean_aps.append(mean_ap)

    ap_0_50, _ = eval_map(bbox_det_results, annotations,
                          iou_thr=0.5, logger='silent')
    ap_0_75, _ = eval_map(bbox_det_results, annotations,
                          iou_thr=0.75, logger='silent')
    return sum(mean_aps) / len(mean_aps), ap_0_50, ap_0_75


def my_bbox_map_eval(det_result):

    # use only bbox det result
    segms = mmcv.concat_list(det_result[1])
    if len(segms) == 0:
        return None, None, None

    segms = mask_util.decode(segms)
    segms = segms.transpose(2, 0, 1)

    bbox_det_result = det_result[0]
    bboxes = np.vstack(bbox_det_result)
    
    # labels = [
    #     np.full(bbox.shape[0], i, dtype=np.int32)
    #     for i, bbox in enumerate(bbox_det_result)
    # ]
    # labels = np.concatenate(labels)

    # return segms, bboxes, labels
    return segms, bboxes


def save_npy(save_dir, split, img_file_name, npy_to_save):

    folder_name = os.path.join(save_dir, split)
    os.makedirs(folder_name, exist_ok=True)
    npy_path = os.path.join(folder_name, img_file_name[:-3] + "npy")
    with open(npy_path, "wb") as f:
        np.save(f, npy_to_save)


class ResultVisualizer(object):
    """Display and save evaluation results.

    Args:
        show (bool): Whether to show the image. Default: True
        wait_time (float): Value of waitKey param. Default: 0.
        score_thr (float): Minimum score of bboxes to be shown.
           Default: 0
    """

    def __init__(self, show=False, wait_time=0, score_thr=0):
        self.show = show
        self.wait_time = wait_time
        self.score_thr = score_thr

    def _save_image_gts_results(self, dataset, results, mAPs, out_dir=None):
        mmcv.mkdir_or_exist(out_dir)

        for mAP_info in mAPs:
            index, mAP = mAP_info
            data_info = dataset.prepare_train_img(index)

            # calc save file path
            filename = data_info['filename']
            if data_info['img_prefix'] is not None:
                filename = osp.join(data_info['img_prefix'], filename)
            else:
                filename = data_info['filename']
            fname, name = osp.splitext(osp.basename(filename))
            save_filename = fname + '_' + str(round(mAP, 3)) + name
            out_file = osp.join(out_dir, save_filename)
            # imshow_gt_det_bboxes(
            #     data_info['img'],
            #     data_info,
            #     results[index],
            #     dataset.CLASSES,
            #     show=self.show,
            #     score_thr=self.score_thr,
            #     wait_time=self.wait_time,
            #     out_file=out_file)

    def evaluate_and_show(self,
                          dataset,
                          results,
                          topk=20,
                          show_dir='work_dir',
                          eval_fn=None):
        """Evaluate and show results.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Det results from test results pkl file
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20
            show_dir (str, optional): The filename to write the image.
                Default: 'work_dir'
            eval_fn (callable, optional): Eval function, Default: None
        """

        assert topk > 0
        if (topk * 2) > len(dataset):
            topk = len(dataset) // 2

        if eval_fn is None:
            eval_fn = bbox_map_eval
        else:
            assert callable(eval_fn)

        prog_bar = mmcv.ProgressBar(len(results))
        _mAPs = {}
        for i, (result, ) in enumerate(zip(results)):
            # self.dataset[i] should not call directly
            # because there is a risk of mismatch
            data_info = dataset.prepare_train_img(i)
            mAP = eval_fn(result, data_info['ann_info'])
            _mAPs[i] = mAP
            prog_bar.update()

        # descending select topk image
        _mAPs = list(sorted(_mAPs.items(), key=lambda kv: kv[1]))
        good_mAPs = _mAPs[-topk:]
        bad_mAPs = _mAPs[:topk]

        good_dir = osp.abspath(osp.join(show_dir, 'good'))
        bad_dir = osp.abspath(osp.join(show_dir, 'bad'))
        self._save_image_gts_results(dataset, results, good_mAPs, good_dir)
        self._save_image_gts_results(dataset, results, bad_mAPs, bad_dir)

    def my_evaluate_and_show(self,
                             dataset,
                             results,
                             show_dir='work_dir',
                             eval_fn=my_bbox_map_eval,
                             threshold=0.5):
        """Evaluate and show results.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Det results from test results pkl file
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20
            show_dir (str, optional): The filename to write the image.
                Default: 'work_dir'
            eval_fn (callable, optional): Eval function, Default: None
        """
        prog_bar = mmcv.ProgressBar(len(results))
        # correct_arry = np.zeros(len(dataset.prepare_train_img(0)[0][0]))
        # gt_array = np.zeros(len(dataset.prepare_train_img(0)[0][0]))
        annotations = []
        for i, (result, ) in enumerate(zip(results)):
            data_info = dataset.prepare_train_img(i)

            # annotations.append(data_info['ann_info'])

            # img_path = os.path.join(
            #     data_info["img_prefix"], data_info["img_info"]['file_name'])
            # img = cv2.imread(img_path)
            try:
                h, w, _ = data_info['img_meta'].data['ori_shape']
            except:
                continue # No cell

            segm_one_map_gt = np.zeros((h, w), dtype=np.int) # (256,256)
            mask = (data_info["gt_masks"]).data # (768,1280)
            gt_labels = data_info["_gt_labels"].data.numpy()

            gt_instance_centers = []

            error_flag = False
            # to obtain an instance_id_map anda mass centers
            for instance_i in range(mask.shape[0]):
                this_mask = cv2.resize(mask[instance_i, :, :], (h, w), interpolation=cv2.INTER_NEAREST)
                indexes_this_instance = np.where(this_mask[:, :] == 1)
                if len(indexes_this_instance[0]) == 0:
                    error_flag = True
                    continue
                segm_one_map_gt[indexes_this_instance[0],
                                indexes_this_instance[1]] = instance_i + 1
                x_max = int(
                    (np.max(indexes_this_instance[1]) + np.min(indexes_this_instance[1])) / 2)
                y_max = int(
                    (np.max(indexes_this_instance[0]) + np.min(indexes_this_instance[0])) / 2)
                gt_instance_centers.append([x_max, y_max])

            if error_flag:
                continue

            save_npy(show_dir, "gt/seg_map",
                     data_info['img_meta'].data['file_name'], segm_one_map_gt)

            # segms, bboxes, labels = eval_fn(result)
            segms, bboxes = eval_fn(result)
            segm_one_map = np.zeros((h, w), dtype=np.int)
            if segms is not None:
                for instance_i in range(segms.shape[0]):
                    indexes_this_instance = np.where(
                        segms[instance_i, :, :] == 1)
                    segm_one_map[indexes_this_instance[0],
                                 indexes_this_instance[1]] = instance_i + 1
            else:
                print(data_info['img_meta'].data['file_name'])

            save_npy(show_dir, "pred/seg_map",
                     data_info['img_meta'].data['file_name'], segm_one_map)

            gt_instance_centers = np.asarray(gt_instance_centers)
            save_npy(show_dir, "gt/instance_centers",
                     data_info['img_meta'].data['file_name'], gt_instance_centers)

            save_npy(show_dir, "gt/instance_types",
                     data_info['img_meta'].data['file_name'], gt_labels)

            # prediction with a threshold
            type_instance_one_map = np.zeros_like((segm_one_map))
            instance_centers = []
            instance_types = []

            if bboxes is not None:
                ind = bboxes[:, -1] > threshold
                bboxes[:,0] = bboxes[:, 0] / (data_info['img_meta'].data['img_shape'][0]/data_info['img_meta'].data['ori_shape'][0])
                bboxes[:,2] = bboxes[:, 2] / (data_info['img_meta'].data['img_shape'][0]/data_info['img_meta'].data['ori_shape'][0])
                bboxes[:,1] = bboxes[:, 1] / (data_info['img_meta'].data['img_shape'][1]/data_info['img_meta'].data['ori_shape'][1])
                bboxes[:,3] = bboxes[:, 3] / (data_info['img_meta'].data['img_shape'][1]/data_info['img_meta'].data['ori_shape'][1])


                bboxes = bboxes[ind]
                # labels = labels[ind]

                for idx, bbox in enumerate(bboxes):
                    instance_centers.append(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                    # instance_types.append(labels[idx])

                    instance_idx = segm_one_map[int(
                        (bbox[1] + bbox[3]) / 2), int((bbox[0] + bbox[2]) / 2)]
                    indexes_this_instance = np.where(
                        segm_one_map == instance_idx)
                    type_instance_one_map[indexes_this_instance] = idx + 1

            instance_centers = np.asarray(instance_centers)
            # instance_types = np.asarray(instance_types)

            save_npy(show_dir, "pred/instance_centers",
                     data_info['img_meta'].data['file_name'], instance_centers)
            # save_npy(show_dir, "pred/instance_types",
            #         data_info["img_info"]['file_name'], instance_types)
            save_npy(show_dir, "pred/seg_map_filtered",
                     data_info['img_meta'].data['file_name'], type_instance_one_map)

            # sec = input('checkpoint data info here.\n')

            prog_bar.update()

        # mAP, ap_0_50, ap_0_75 = bbox_map_eval_total(results, annotations)
        # print("mAP: ", mAP)
        # print("ap 0.5: ", ap_0_50)
        # print("ap 0.75: ", ap_0_75)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test pkl result')
    parser.add_argument(
        'show_dir', help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=0,
        help='the interval of show (s), 0 is block')
    parser.add_argument(
        '--topk',
        default=20, 
        type=int,
        help='saved Number of the highest topk '
        'and lowest topk after index sorting')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0,
        help='score threshold (default: 0.)')
    # parser.add_argument(
    #     '--cfg-options',
    #     nargs='+',
    #     action=DictAction,
    #     help='override some settings in the used config, the key-value pair '
    #     'in xxx=yyy format will be merged into config file. If the value to '
    #     'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
    #     'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    #     'Note that the quotation marks are necessary and that no white space '
    #     'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mmcv.check_file_exist(args.prediction_path)

    cfg = Config.fromfile(args.config)
    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.with_mask = True
    cfg.data.test.test_mode = True
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # cfg.data.test.pop('samples_per_gpu', 0)
    # cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.prediction_path)

    result_visualizer = ResultVisualizer(
        args.show, args.wait_time, args.show_score_thr)
    result_visualizer.my_evaluate_and_show(
        dataset, outputs, show_dir=args.show_dir)


if __name__ == '__main__':
    main()
