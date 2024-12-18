from collections import Sequence

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import cv2
from scipy.spatial import distance


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.

    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".

    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(min(img_scale_long),
                                          max(img_scale_long) + 1)
            short_edge = np.random.randint(min(img_scale_short),
                                           max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale


def show_ann(coco, img, ann_info):
    plt.imshow(mmcv.bgr2rgb(img))
    plt.axis('off')
    coco.showAnns(ann_info)
    plt.show()


def points_mask_iou(points, mask_gt):
    '''
    input:
    points：采样轮廓点
    mask_gt:实际掩码
    output:
    IOU
    '''
    polar_mask = np.zeros(mask_gt.shape)
    polar_mask = cv2.fillPoly(polar_mask,
                              [points.cpu().numpy().astype(np.int64)],
                              (255, 255, 255))
    polar_mask = polar_mask > 0

    if mask_gt.max() > 1:
        mask_gt = mask_gt > 0
    intersection = (polar_mask * mask_gt).sum()
    union = polar_mask.sum() + mask_gt.sum() - intersection
    iou = intersection / union
    return iou


def get_sample_contour_idx(all_contour_points, num_sample_contour_points):
    length_all_contour_points = len(all_contour_points)
    sample_interval = round(length_all_contour_points /
                            num_sample_contour_points)
    sample_interval = max(1, sample_interval)
    sample_contour_idx = np.arange(0,
                                   num_sample_contour_points * sample_interval,
                                   sample_interval)

    if sample_contour_idx[-1] >= length_all_contour_points:
        sample_contour_idx = sample_contour_idx % length_all_contour_points
        sample_contour_idx.sort()
    return sample_contour_idx


def contour_random_walk(explore_times,
                        num_sample_points,
                        num_steps,
                        step,
                        all_contour_points,
                        mask,
                        extreme_included,
                        start=0,
                        show_iou=True):
    '''
    先对all_contour_points的顺序进行调整，使传入的start为数组的第一个点
    return:
    all_contour_points[best_sample_idx, :] 尺寸为(N,2),坐标为[y,x]
    '''
    all_contour_points = torch.cat(
        [all_contour_points[start:], all_contour_points[:start]])
    if extreme_included:
        # 在轮廓上先取4个极值点，再取等间隔的32个点，并按照沿边缘的顺序排列好
        # y_max = all_contour_points[:, 0].max()  # 坐标
        # y_min = all_contour_points[:, 0].min()  # 坐标
        # x_max = all_contour_points[:, 1].max()  # 坐标
        # x_min = all_contour_points[:, 1].min()  # 坐标
        y_max_id = all_contour_points[:, 0].argmax()
        y_min_id = all_contour_points[:, 0].argmin()
        x_max_id = all_contour_points[:, 1].argmax()
        x_min_id = all_contour_points[:, 1].argmin()
        extreme_idx = [y_max_id, y_min_id, x_max_id, x_min_id]
        other_points_idx = np.array(
            list(set(range(len(all_contour_points))) - set(extreme_idx)))
        other_points = all_contour_points[other_points_idx]
        other_sample_idx_idx = get_sample_contour_idx(other_points,
                                                      32)  # 只剩下32个点 ,array
        other_sample_idx = other_points_idx[other_sample_idx_idx]
        all_sample_idx = other_sample_idx.tolist() + extreme_idx
        best_sample_idx = sorted(all_sample_idx)
        sample_points = all_contour_points[best_sample_idx]

    else:
        best_sample_idx = get_sample_contour_idx(all_contour_points,
                                                 num_sample_points)
        sample_points = all_contour_points[best_sample_idx, :]

    if explore_times > 0:
        best_iou = points_mask_iou(sample_points, mask)
    for i in range(explore_times):
        sample_idx = best_sample_idx.copy()
        for j in range(num_steps):
            forward_idx = np.random.choice(len(sample_points),
                                           int(num_sample_points * 0.5))
            sample_idx[forward_idx] += step
            sample_idx = np.clip(sample_idx,
                                 a_min=0,
                                 a_max=len(all_contour_points) - 1)
            sample_points = all_contour_points[sample_idx, :]
            iou = points_mask_iou(sample_points, mask)

            if iou > best_iou:
                best_iou = iou
                best_sample_idx = sample_idx.copy()
    # print('sample contour points,',num_sample_points,'points',best_iou)
    if show_iou:
        return best_iou, all_contour_points[best_sample_idx, :]
    else:
        return all_contour_points[best_sample_idx, :]


def fillInstance(instance, center_fill_before):
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
        # 记录轮廓上的每个点的轮廓序号和点号
        for i in range(len(contours)):
            for j in range(contours[i].shape[0]):
                e_x = str(contours[i][j][0][0])
                e_y = str(contours[i][j][0][1])
                dictEdgePoint[e_x + "_" + e_y] = [i, j
                                                  ]  # 像素(e_x,e_y)是第i条轮廓的第j个轮廓点

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
            if index_y > distanceMapDown[index_x]:  # index_y
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
                selected_info[str(e_x) + "_" + str(e_y)] = dictEdgePoint[
                    str(e_x) + "_" + str(e_y)]  # 与上一个循环中放入的点会有重复
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
                               key=lambda x: (x[1], x[0]))  # 先按照轮廓序号、轮廓点序号排序

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
                                          len(path) - 1)  # start始终不为0，第一个点保持不变
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
                if compare_path(best_path, path, distMatrix):
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
    if center_fill_before:
        return instance, (center_x, center_y)
    else:
        return instance


def compare_path(path_1, path_2, distMatrix):
    sum1 = 0
    for i in range(1, len(path_1)):
        sum1 += distMatrix[path_1[i - 1]][path_1[i]]

    sum2 = 0
    for i in range(1, len(path_2)):
        sum2 += distMatrix[path_2[i - 1]][path_2[i]]

    return sum1 > sum2