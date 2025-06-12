import numpy as np


def f1_score(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    # dice = np.mean(2*intersect/total_sum)
    dice = round((2*intersect/total_sum), 3)
    # return round(dice, 3) #round up to 3 decimal places
    return dice

def iou_score(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    # iou = np.mean(intersect/union)
    iou = round((intersect/union),3)
    return iou
    # return round(iou, 3)