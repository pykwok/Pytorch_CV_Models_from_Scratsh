import torch

from collections import Counter

def intersection_over_union(boxes_preds, # (BATCH_SIZE, 4)
                            boxes_gt,    # (BATCH_SIZE, 4)
                            box_format = "midpoint"): # (center_x, center_y, w, h) or (x1, y1, x2, y2)
    """
        Calculates intersection over union
        Parameters:
            boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
            boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
            box_format (str): midpoint/corners, if boxes (center_x, center_y, w, h) or (x1, y1, x2, y2)
        Returns:
            tensor: Intersection over union for all examples
        """
    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape

    #------------------------------------------------------------
    # 一、取出 两个框的 左上角 和 右下角 的数值，一共 4 + 4 = 8 个数
    #------------------------------------------------------------
    if box_format == "midpoint": # (center_x, center_y, w, h)
        box_preds_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box_preds_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box_preds_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box_preds_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box_gt_x1 = boxes_gt[..., 0:1] - boxes_gt[..., 2:3] / 2
        box_gt_y1 = boxes_gt[..., 1:2] - boxes_gt[..., 3:4] / 2
        box_gt_x2 = boxes_gt[..., 0:1] + boxes_gt[..., 2:3] / 2
        box_gt_y2 = boxes_gt[..., 1:2] + boxes_gt[..., 3:4] / 2

    elif box_format == "corners": # (x1, y1, x2, y2)
        box_preds_x1 = boxes_preds[..., 0:1]
        box_preds_y1 = boxes_preds[..., 1:2]
        box_preds_x2 = boxes_preds[..., 2:3]
        box_preds_y2 = boxes_preds[..., 3:4]

        box_gt_x1 = boxes_gt[..., 0:1]
        box_gt_y1 = boxes_gt[..., 1:2]
        box_gt_x2 = boxes_gt[..., 2:3]
        box_gt_y2 = boxes_gt[..., 3:4]

    # ------------------------------------------------------------
    # 二、根据 之前的8个数，算出 交集矩阵的 (x1, y1, x2, y2)
    # ------------------------------------------------------------

    # 防止两个框没有交集： 用clamp(0) 函数
    x1 = torch.max(box_preds_x1, box_gt_x1)
    y1 = torch.max(box_preds_y1, box_gt_y1)
    x2 = torch.min(box_preds_x2, box_gt_x2)
    y2 = torch.min(box_preds_y2, box_gt_y2)

    # ------------------------------------------------------------
    # 三、 IOU = 交集 / 并集
    # ------------------------------------------------------------

    intersection_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    boxes_preds_area = abs((box_preds_x2 - box_preds_x1) * (box_preds_y2 - box_preds_y1))
    boxes_gt_area = abs((box_gt_x2 - box_gt_x1) * (box_gt_y2 - box_gt_y1))

    union_area =  boxes_preds_area + boxes_gt_area - intersection_area + 1e-6

    iou = intersection_area / union_area

    return iou


def nms(box_preds, # [[class, probability_score, x1, y1, x2, y2], ... ]
        iou_threshold,
        prob_threshold,
        box_format = "corners"
        ):
    assert type(box_preds) == list

    # bbox_preds 的内容 [[class, probability, x1, y1, x2, y2], ... ]

    # 置信度比较低的 预测框 不要
    bboxes = [box for box in box_preds if box[1] > prob_threshold]

    # 按照置信度，从高到低排序
    bboxes = sorted(bboxes,
                    key = lambda x:x[1],
                    reverse=True)

    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        #1. 如果 box[0] != chosen_box[0] ：种类不同，可以留下这个 预测框
        #               ==               ：种类相同

        #2. 如果 IOU(置信度最高的框，同一种类的其它框) > IOU阈值，不要
        #                                          < IOU阈值，可以留下
        boxes = [box
                 for box in bboxes
                 if box[0] != chosen_box[0]
                 or intersection_over_union(torch.tensor(chosen_box[2:]),
                                            torch.tensor(box[2:]),
                                            box_format = box_format) < iou_threshold
                 ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

# 这个函数是针对某一个IOU阈值。举例： `mAP@.5`
# 不是 `mAP@0.5:0.05:0.95`
def mean_avg_precision(pred_boxes, # [train_idx, class, prob_score, x1, y1, x2, y2]
                       true_boxes,
                       iou_threshold = 0.5,
                       box_format = "midpoint",
                       num_class = 20
                       ):
    """
        Calculates mean average precision
        Parameters:
            pred_boxes (list): list of lists containing all bboxes with each bboxes
            specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
            true_boxes (list): Similar as pred_boxes except all the correct ones
            iou_threshold (float): threshold where predicted bboxes is correct
            box_format (str): "midpoint" or "corners" used to specify bboxes
            num_classes (int): number of classes
        Returns:
            float: mAP value across all classes given a specific IoU threshold
        """

    average_precision = []
    epsilon = 1e-6


























