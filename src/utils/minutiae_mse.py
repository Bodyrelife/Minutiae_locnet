import numpy as np
import scipy.io as scio
import show_minutiae
from math import pi, sqrt
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean as eucl_dis
pi2 = 2 * pi

def load_pre_boxes(matfile):
    pre_boxes = scio.loadmat(matfile)['aboxes'][0][0]
    # convert to dict
    pre_boxes_dict = {}
    for box in pre_boxes:
        pre_boxes_dict[box[1][0]] = box[0]
    return pre_boxes_dict

def delta_ori(ori1, ori2):
    ori1 %= 2 * pi
    ori2 %= 2 * pi
    delta_ori = abs(ori1 - ori2)
    return min(delta_ori, 2 * pi - delta_ori)

MAX_DIS, MAX_ORI = 15, pi / 6
# KDTree version
def find_relationship(boxes, gt_boxes, one2one_mapping=False):
    boxes, gt_boxes = np.array(boxes), np.array(gt_boxes)
    boxes_tree = KDTree(boxes[:, :2])
    #                   loc_dis  ori_dis gt_idx
    boxes2gt = np.array([[None, None, None]] * len(boxes))
    recall_minutiae = 0
    for idx, gt in enumerate(gt_boxes):
        valid_boxes = np.array(boxes_tree.query_ball_point(gt[:2], MAX_DIS))
        valid_boxes = filter(lambda x:delta_ori(boxes[x][2], gt[2])<MAX_ORI, valid_boxes)
        valid_boxes = [[eucl_dis(boxes[i][:2], gt[:2]), delta_ori(boxes[i][2], gt[2]), i] \
                        for i in valid_boxes]
        valid_boxes.sort(key=lambda x:x[0])
        if len(valid_boxes) > 0: recall_minutiae += 1 
        for i, box in enumerate(valid_boxes):
            if one2one_mapping and i > 0:
                break
            if box[0] < boxes2gt[box[2]][0] or boxes2gt[box[2]][0] is None:
                 boxes2gt[box[2]][0] = box[0]
                 boxes2gt[box[2]][1] = box[1]
                 boxes2gt[box[2]][2] = idx
    return boxes2gt

# Distance matrix version
def find_relationship2(boxes, gt_boxes):
    boxes, gt_boxes = np.array(boxes), np.array(gt_boxes)
    dis = cdist(boxes[:, :2], gt_boxes[:, :2])
    mindis, idx = np.min(dis, axis=1), np.argmin(dis, axis=1)
    angle = abs(boxes[:,2] - gt_boxes[idx,2])
    angle = np.min(np.array([angle, pi2-angle]), axis=0)  
    label = np.array([mindis <= MAX_DIS, angle < MAX_ORI]).all(axis=0)
    boxes2gt = np.array([[None, None, None]] * len(boxes))
    boxes2gt[label, 0] = mindis[label]
    boxes2gt[label, 1] = angle[label]
    boxes2gt[label, 2] = idx[label]
    return boxes2gt

def evaluation(boxes, gt_boxes):
    count_correct, count_detect = 0, 0
    ave_recall, count_groundtruth = 0, 0
    ave_position_var, ave_ori_var = 0, 0
    ave_precision_unique = 0
    for i, pic_key in enumerate(gt_boxes.keys()):
        if pic_key not in boxes.keys(): continue
        if len(gt_boxes[pic_key]) == 0: continue
        if len(boxes[pic_key]) > 0:
            boxes2gt = find_relationship(
                boxes[pic_key], gt_boxes[pic_key])
        else:
            boxes2gt = np.array([])
        count_detect += len(boxes2gt)
        boxes2gt = filter(lambda x:x[0] is not None, boxes2gt)
        count_correct += len(boxes2gt)
        ave_recall += len(set([x[2] for x in boxes2gt]))
        count_groundtruth += len(gt_boxes[pic_key])
        ave_position_var += sum([x[0] for x in boxes2gt])
        ave_ori_var += sum([x[1] for x in boxes2gt])
    print count_detect, count_correct, ave_recall, count_groundtruth
    if count_correct != 0:
        ave_ori_var = float(ave_ori_var) / count_correct / pi * 180
        ave_position_var = float(ave_position_var) / count_correct    
    if count_detect != 0:
        ave_precision_unique = float(ave_recall) / count_detect
        ave_precision = float(count_correct) / count_detect
    if count_groundtruth != 0:
        ave_recall = float(ave_recall) / count_groundtruth
    return ave_recall, ave_precision_unique, ave_precision, ave_position_var, ave_ori_var