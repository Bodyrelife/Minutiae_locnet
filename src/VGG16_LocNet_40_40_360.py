import os
import numpy as np
from math import pi
image_batch_size = 2
window_batch_size = 128
assert window_batch_size % image_batch_size == 0

pretrained_model = os.path.join('..', 'model-defs', 'pre_trained_models', 'vgg_16layers', 'vgg16.caffemodel')
mean_file = os.path.join('..', 'model-defs', 'pre_trained_models', 'vgg_16layers', 'mean_image.mat')
solver_file = os.path.join('..', 'model-defs', 'VGG16_LocNet_40_40_360', 'solver.prototxt')
deploy_file = os.path.join('..', 'model-defs', 'VGG16_LocNet_40_40_360', 'deploy.prototxt')

# load proposals from faster rcnn results
from utils.minutiae_mse import load_pre_boxes
FASTER_PRO = os.path.join('.', 'nist27_4205_preboxes.mat')
pre_boxes = load_pre_boxes(FASTER_PRO)

from utils.common import r_int, gen_rand_shift
from utils.minutiae_mse import find_relationship, find_relationship2
def wh2tlbr(wh, shape):
    w, h = wh
    tlbr = [max(w-19, 0), max(h-19, 0), \
        min(w+20, shape[1]), min(h+20, shape[0])]
    return tlbr

def tlbr2wh(tlbr):
    tlw, tlh, brw, brh = tlbr
    w = r_int((tlw + brw - 1) / 2.)
    h = r_int((tlh + brh - 1) / 2.)
    return (w, h)

def regions_loc2who(pred_loc, regions_loc):
    who = []
    for i in xrange(len(pred_loc)):     
        pre_w = pred_loc[i, :40] 
        pre_h = pred_loc[i, 40:80]
        pre_o = pred_loc[i, 80:]
        if max(pre_w) < 0 or max(pre_h) < 0:
            who.append([None, None, None])
        else: # to do analysis
            d_w = np.asarray(pre_w).argmax() - 19
            d_h = np.asarray(pre_h).argmax() - 19
            w, h = tlbr2wh(regions_loc[i, 1:5])
            o = np.asarray(pre_o).argmax() / 180. * pi
            who.append([w + d_w, h + d_h, o])
    who = filter(lambda x:x[0] is not None, who)
    return who

def gen_label_lossweight(d_w, d_h, o):
    M = 40
    base=np.array([19, 59, 80])
    label, loss = np.zeros(440), np.ones(440) * 0.5 * M / (M - 1)
    place = np.array([d_w, d_h, int(o/pi*180)]) + base
    label[place], loss[place] = 1, 0.5 * M 
    return label, loss

def gen_pos_neg_bbox(pic_key, ground_truth, shape, batch_idx, ifneg=False, ifrand=False):
    process_num, total_num = 0, window_batch_size / image_batch_size
    regions_loc, label = np.zeros([total_num, 9]), np.zeros([total_num, 440])
    #---------------------------------------------------
    loss_weight = np.ones([total_num, 440])
    #---------------------------------------------------
    regions_loc[:, 0] = batch_idx
    if pic_key in pre_boxes.keys():
        box = pre_boxes[pic_key]
        pre_pick = np.random.choice(len(box), min(total_num, len(box)), False)
        boxes2gt = find_relationship(pre_boxes[pic_key], ground_truth)
        for pre_idx in pre_pick:
            if boxes2gt[pre_idx][0] is not None:
                regions_loc[process_num, 1:5] = wh2tlbr(box[pre_idx, :2], shape)
                w, h, o = ground_truth[boxes2gt[pre_idx][2]]
                d_w, d_h = r_int(w - box[pre_idx, 0]), r_int(h - box[pre_idx, 1])
                label[process_num, :], loss_weight[process_num, :] = \
                    gen_label_lossweight(d_w, d_h, o)
            elif ifneg:
                regions_loc[process_num, 1:5] = wh2tlbr(box[pre_idx, :2], shape)
            process_num += 1

    max_pos_shift = 14
    gd_pick = np.random.choice(len(ground_truth), total_num - process_num)
    if ifrand:
        for gd_idx in gd_pick:
            d_w, d_h = gen_rand_shift(max_pos_shift)
            w, h, o = ground_truth[gd_idx]
            w, h = w - d_w, h - d_h
            regions_loc[process_num, 1:5] = wh2tlbr([w, h], shape)
            #        label for w       label for h       label for o
            label[process_num, :], loss_weight[process_num, :] = \
                gen_label_lossweight(d_w, d_h, o)
            process_num += 1
    #----------------------------------------
    regions_loc = regions_loc[:process_num, :]
    loss_weight = loss_weight[:process_num, :]
    label = label[:process_num, :]
    loss_weight[:, 80:] = 0
    #----------------------------------------
    return regions_loc.tolist(), label.tolist(), loss_weight.tolist()

if __name__ == '__main__':
    label, loss = gen_label_lossweight(-8, -5, 255)
    regions_loc[process_num, 1:5] = wh2tlbr([w, h], shape)
