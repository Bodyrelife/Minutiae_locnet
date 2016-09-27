import numpy as np
import scipy.io as scio
import show_minutiae
from math import pi, sqrt
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean as eucl_dis

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
    return boxes2gt, recall_minutiae

def main():
    preboxes = '../nist27_4205_preboxes.mat'
    boxes = load_pre_boxes(preboxes)
    print cal_mse(boxes[0, 0], boxes[1, 0])

if __name__ == '__main__':
    main()