import torch

from mmdet.ops.nms import nms_wrapper
from IPython import embed


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels


# original
# def multiclass_nms_with_mask(multi_bboxes,
#                    multi_scores,
#                    multi_masks,
#                    score_thr,
#                    nms_cfg,
#                    max_num=-1,
#                    score_factors=None):
#     """NMS for multi-class bboxes.

#     Args:
#         multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
#         multi_scores (Tensor): shape (n, #class)
#         score_thr (float): bbox threshold, bboxes with scores lower than it
#             will not be considered.
#         nms_thr (float): NMS IoU threshold
#         max_num (int): if there are more than max_num bboxes after NMS,
#             only top max_num will be kept.
#         score_factors (Tensor): The factors multiplied to scores before
#             applying NMS

#     Returns:
#         tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
#             are 0-based.
#     """
#     num_classes = multi_scores.shape[1]
#     bboxes, labels, masks = [], [], []
#     nms_cfg_ = nms_cfg.copy()
#     nms_type = nms_cfg_.pop('type', 'nms')
#     nms_op = getattr(nms_wrapper, nms_type)
#     for i in range(1, num_classes):
#         cls_inds = multi_scores[:, i] > score_thr
#         if not cls_inds.any():
#             continue
#         # get bboxes and scores of this class
#         if multi_bboxes.shape[1] == 4:
#             _bboxes = multi_bboxes[cls_inds, :]
#             _masks  = multi_masks[cls_inds, :]
#         else:
#             _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
#         _scores = multi_scores[cls_inds, i]
#         if score_factors is not None:
#             _scores *= score_factors[cls_inds]
#         cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
#         cls_dets, index = nms_op(cls_dets, **nms_cfg_)
#         cls_masks = _masks[index]
#         cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
#                                            i - 1,
#                                            dtype=torch.long)
#         bboxes.append(cls_dets)
#         labels.append(cls_labels)
#         masks.append(cls_masks)
#     if bboxes:
#         bboxes = torch.cat(bboxes)
#         labels = torch.cat(labels)
#         masks = torch.cat(masks)
#         if bboxes.shape[0] > max_num:
#             _, inds = bboxes[:, -1].sort(descending=True)
#             inds = inds[:max_num]
#             bboxes = bboxes[inds]
#             labels = labels[inds]
#             masks = masks[inds]
#     else:
#         bboxes = multi_bboxes.new_zeros((0, 5))
#         labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
#         masks = multi_bboxes.new_zeros((0, 2, 36))

#     return bboxes, labels, masks


def multiclass_nms_with_mask(multi_bboxes,
                             multi_scores,
                             multi_masks,
                             score_thr,
                             nms_cfg,
                             max_num=-1,
                             score_factors=None,
                             vis_info=dict({})):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    #########
    if len(vis_info) == 3:
        multi_center, multi_radius, multi_angle = vis_info['center'], vis_info[
            'radius'], vis_info['angle']
    elif len(vis_info) == 1:
        multi_center = vis_info['center']
    ########
    

    # slnms, add by amanda
    multi_scores = torch.max(multi_scores, 1)[0][:, None]
    multi_scores = torch.cat([torch.ones(multi_scores.shape).cuda(), multi_scores], 1)
    num_classes = multi_scores.shape[1]


    # bboxes, labels, masks = [], [], []
    bboxes, labels, masks, centers, radius, angles = [], [], [], [], [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    

    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
            _masks = multi_masks[cls_inds, :]
            ######
            if len(vis_info) == 3:
                _centers = multi_center[cls_inds, :]
                _radius = multi_radius[cls_inds, :]
                _angles = multi_angle[cls_inds, :]
            elif len(vis_info) == 1:
                _centers = multi_center[cls_inds, :]
            ########

        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]

        cls_dets = torch.cat([_bboxes, _scores[:, None]],
                             dim=1)  # confidence of class

        cls_dets, index = nms_op(cls_dets, **nms_cfg_)
        cls_masks = _masks[index]  # 按照分类的分数选mask
        ####
        if len(vis_info) == 3:
            cls_centers = _centers[index]
            cls_radius = _radius[index]
            cls_angles = _angles[index]
        elif len(vis_info) == 1:
            cls_centers = _centers[index]
        ####
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
        masks.append(cls_masks)

        #######
        if len(vis_info) == 3:
            centers.append(cls_centers)
            radius.append(cls_radius)
            angles.append(cls_angles)
        elif len(vis_info) == 1:
            centers.append(cls_centers)
        #######

    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        masks = torch.cat(masks)
        ####
        if len(vis_info) == 3:
            centers = torch.cat(centers)
            radius = torch.cat(radius)
            angles = torch.cat(angles)
        elif len(vis_info) == 1:
            centers = torch.cat(centers)
        ####
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
            masks = masks[inds]
            ######
            if len(vis_info) == 3:
                centers = centers[inds]
                radius = radius[inds]
                angles = angles[inds]
            elif len(vis_info) == 1:
                centers = centers[inds]
            ######
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        masks = multi_bboxes.new_zeros((0, 2, 36))
        #############
        if len(vis_info) == 3:
            centers = multi_bboxes.new_zeros((0, 2))
            radius = multi_bboxes.new_zeros((0, 36))
            angles = multi_bboxes.new_zeros((0, 36))
        elif len(vis_info) == 1:
            centers = multi_bboxes.new_zeros((0, 2))
        ############

    if len(vis_info) == 3:
        return bboxes, labels, masks, centers, radius, angles
    elif len(vis_info) == 1:
        return bboxes, labels, masks, centers
    else:
        return bboxes, labels, masks
