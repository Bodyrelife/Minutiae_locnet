#coding=utf-8 
import os
import sys
import glob
import cv2
import random as rand
import scipy.io as scio
sys.path.append(os.path.join('..', 'external', 'caffe_LocNet', 'python'))
import caffe
import numpy as np
import logging  
from datetime import datetime

GPU_ID = 0
OUTPUT_DIR = os.path.join('..', 'output', datetime.now().strftime('%Y%m%d-%H:%M'))
FASTER_PRO = os.path.join('.', 'nist27_4205_preboxes.mat')

from utils.common import init_log
logging = init_log(OUTPUT_DIR)

logging.info('Train Test Settings: VGG16_LocNet_40_40_360')
from models.VGG16_LocNet_40_40_360 import \
    image_batch_size, window_batch_size, \
    pretrained_model, solver_file, deploy_file
assert window_batch_size % image_batch_size == 0

# load proposals from faster rcnn results
from utils.minutiae_mse import load_pre_boxes
pre_boxes = load_pre_boxes(FASTER_PRO)

# image mean
mean_pic = [0.48501959, 0.45795685, 0.40760392]


from math import pi, cos, sin
def gen_rand_shift(max_shift, min_shift = 0):
    rand_shift = rand.random() * max_shift + min_shift
    rand_angle = rand.random() * 2 * pi
    rand_w_h = np.asarray([round(rand_shift * cos(rand_angle)), \
        round(rand_shift * sin(rand_angle))])
    return [int(x) for x in rand_w_h]

from scipy.spatial import KDTree
from utils.common import r_int, wh2tlbr, tlbr2wh
from utils.minutiae_mse import find_relationship
def gen_pos_neg_bbox(pic_key, ground_truth, shape, batch_idx):
    # gen pos, which contains a minutiae, neg half to half
    # currently random gen neg, and neg sample's orentiation loss weight = 0
    # could be results from faster-rcnn
    process_num, total_num = 0, window_batch_size / image_batch_size
    regions_loc, label = np.zeros([total_num, 9]), np.zeros([total_num, 440])
    loss_weight = np.ones([total_num, 440])
    regions_loc[:, 0] = batch_idx

    if pic_key in pre_boxes.keys():
        box = pre_boxes[pic_key]
        pre_pick = np.random.choice(len(box), min(total_num, len(box)), False)
        boxes2gt, _ = find_relationship(pre_boxes[pic_key], ground_truth, False)
        for i, pre_idx in enumerate(pre_pick):
            regions_loc[i, 1:5] = wh2tlbr(box[pre_idx, :2], shape)
            if boxes2gt[pre_idx][0] is not None:
                w, h, o = ground_truth[boxes2gt[pre_idx][2]]
                d_w, d_h = r_int(w - box[pre_idx, 0]), r_int(h - box[pre_idx, 1])
                label[i, [19+d_w, 59+d_h, 80+int(o / pi * 180)]] = 1
            else:
                loss_weight[i, 80:] = 0
        process_num = len(pre_pick)
    max_pos_shift = 15
    gd_pick = np.random.choice(len(ground_truth), total_num - process_num)
    for i, gd_idx in enumerate(gd_pick):
        rand_w_h = gen_rand_shift(max_pos_shift)
        w, h, o = ground_truth[gd_idx]
        w, h = np.asarray([w, h]) + rand_w_h
        regions_loc[i+process_num, 1:5] = wh2tlbr([w, h], shape)
        #        label for w       label for h       label for o
        label[i+process_num, [19+rand_w_h[0], 59+rand_w_h[1], 80+int(o / pi * 180)]] = 1
    return regions_loc.tolist(), label.tolist(), loss_weight.tolist()

from utils.show_minutiae import show_mnt
def data_prepare(solver, train_set, train_sample_rate):
    # random pick batch_size images from dataset
    set_pick = np.random.choice(len(train_set), 1, p=train_sample_rate)[0]
    pic_pick = np.random.choice(len(train_set[set_pick]), \
        image_batch_size, replace=False)
    # load image
    images, minutiae = [], []
    regions_loc, label, loss_weight = [], [], []
    for i, idx in enumerate(pic_pick):
        pic_key = train_set[set_pick].keys()[idx]
        # prepare image data
        I = cv2.imread(pic_key)
        r_l, l, l_w = gen_pos_neg_bbox(pic_key, train_set[set_pick][pic_key], I.shape, i)
        I = np.asarray(cv2.imread(pic_key), dtype=np.float32) / 255
        for c in xrange(I.shape[2]):
            I[:, :, c] -= mean_pic[c]
        I = I.transpose((2, 0, 1))
        images.append(I)
        regions_loc.extend(r_l)
        label.extend(l)
        loss_weight.extend(l_w)
    # prepare batch
    images = np.asarray(images)
    regions_loc = np.asarray(regions_loc)
    label = np.asarray(label)
    loss_weight = np.asarray(loss_weight) 
    solver.net.blobs['data'].reshape(*images.shape)
    solver.net.blobs['data'].data[...] = images
    solver.net.blobs['regions_loc'].reshape(*regions_loc.shape)
    solver.net.blobs['regions_loc'].data[...] = regions_loc    
    solver.net.blobs['label'].reshape(*label.shape)
    solver.net.blobs['label'].data[...] = label
    solver.net.blobs['loss_weights'].reshape(*loss_weight.shape)
    solver.net.blobs['loss_weights'].data[...] = loss_weight    
    solver.net.reshape()

from caffe.proto import caffe_pb2
import google.protobuf as pb2
def get_info_from_solver(solver_file=solver_file):
    solver_param = caffe_pb2.SolverParameter()
    with open(solver_file, 'rt') as f:
        pb2.text_format.Merge(f.read(), solver_param)
    return solver_param

def model_prepare():
    logging.info('Prepare Model ...')
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
    solver = caffe.get_solver(solver_file)
    solver.net.copy_from(pretrained_model)
    logging.info('Done.')
    return solver

def loc2who(pred_loc, regions_loc):
    who = []
    for i in xrange(len(pred_loc)):
        pre_w = pred_loc[i][:40] 
        pre_h = pred_loc[i][40:80]
        pre_o = pred_loc[i][80:]
        if max(pre_w) < 0.5 or max(pre_h) < 0.5:
            who.append([None, None, None])
        else: # to do analysis
            d_w = np.asarray(pre_w).argmax() - 19
            d_h = np.asarray(pre_h).argmax() - 19
            w, h = tlbr2wh(regions_loc[i][1:5])
            o = np.asarray(pre_o).argmax() / 180. * pi
            who.append([w + dw, h + dh, o])
    return who

def evaluation(boxes, gt_boxes):
    logging.info('Evaluating dataset %d:'%(len(gt_boxes)))
    precision_ave, precision_all = 0, 0
    recall_ave, recall_all = 0, 0
    loc_ave, ori_ave = 0, 0
    for i, pic_key in enumerate(gt_boxes.keys()):
        if pic_key not in boxes.keys(): continue
        if len(gt_boxes[pic_key]) == 0: continue
        if len(boxes[pic_key]) > 0:
            boxes2gt, recall_minutiae = find_relationship(
                boxes[pic_key], gt_boxes[pic_key])
        else:
            boxes2gt, recall_minutiae = [], 0
        precision_all += len(boxes2gt)
        boxes2gt = filter(lambda x:x[0] is not None, boxes2gt)
        precision_ave += len(boxes2gt)
        recall_ave += recall_minutiae
        recall_all += len(gt_boxes[pic_key])
        loc_ave += sum([x[0] for x in boxes2gt])
        ori_ave += sum([x[1] for x in boxes2gt])
    if precision_ave != 0:
        ori_ave = float(ori_ave) / precision_ave / pi * 180
        loc_ave = float(loc_ave) / precision_ave    
    if precision_all != 0:
        precision_ave = float(precision_ave) / precision_all
    if recall_all != 0:
        recall_ave = float(recall_ave) / recall_all
    logging.info('Rec: %.3f, Pre: %.3f, Loc: %.3f, Ori: %.3f' \
        %(recall_ave, precision_ave, loc_ave, ori_ave))

def model_test(solver, data_set):
    loss, pred_res  = 0, {}
    for i, pic_key in enumerate(data_set.keys()):
        logging.info('Testing: %s'%(pic_key))
        # prepare image data
        I = cv2.imread(pic_key)
        r_l, l, l_w = gen_pos_neg_bbox(pic_key, data_set[pic_key], I.shape, 0)
        I = np.asarray(cv2.imread(pic_key), dtype=np.float32) / 255
        for c in xrange(I.shape[2]):
            I[:, :, c] -= mean_pic[c]
        I = I.transpose((2, 0, 1))
        # prepare batch
        images = np.asarray([I])
        regions_loc = np.asarray(r_l)
        label = np.asarray(l)
        loss_weight = np.asarray(l_w) 
        solver.net.blobs['data'].reshape(*images.shape)
        solver.net.blobs['data'].data[...] = images
        solver.net.blobs['regions_loc'].reshape(*regions_loc.shape)
        solver.net.blobs['regions_loc'].data[...] = regions_loc    
        solver.net.blobs['label'].reshape(*label.shape)
        solver.net.blobs['label'].data[...] = label
        solver.net.blobs['loss_weights'].reshape(*loss_weight.shape)
        solver.net.blobs['loss_weights'].data[...] = loss_weight    
        solver.net.reshape()
        solver.net.forward()
        loss_temp = solver.net.blobs['loss_loc'].data
        loss += loss_temp
        predwho = loc2who(solver.net.blobs['preds_loc'].data, regions_loc)
        predwho = filter(lambda x:x[0] is not None, predwho)
        pred_res[pic_key] = predwho
        logging.info('loss: %f ...Done!'%(loss_temp))
    logging.info('Average loss: %f'%(loss / len(data_set)))
    evaluation(pred_res, data_set)

from utils.common import moving_average
from load_data import train_valid_test_split
def model_train():
    train_set, valid_set, test_set, train_sample_rate = \
        train_valid_test_split(logging)
    params = get_info_from_solver()
    solver, loss = model_prepare(), None
    for _iter in xrange(params.max_iter):
        # prepare training data here need to be multiprocessing in future
        data_prepare(solver, train_set, train_sample_rate)
        # train model here and get the loss
        solver.step(1)
        loss = moving_average(solver.net.blobs['loss_loc'].data, loss)
        if solver.iter % 10 == 0:
            logging.info('Iter: step %d loss: %.4f' %(solver.iter, loss))

        if solver.iter % 100 == 0:
            # model_test(solver, test_set[0])
            pass

        if solver.iter % params.snapshot == 0:
            logging.info('Saving snapshothot {0} ...'.format(_iter))
            filename = params.snapshot_prefix + \
                '_iter_{0}'.format(solver.iter) + '.caffemodel'
            filename = os.path.join(OUTPUT_DIR, filename)
            solver.net.save(filename)
            logging.info('Done.')


def train():
    model_train()
    train_set, valid_set, test_set, train_sample_rate = \
        train_valid_test_split(logging)
    # evaluation(pre_boxes, test_set[0])

if __name__ == '__main__':
    train()