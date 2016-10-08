#coding=utf-8 
import os
import sys
import time
import cv2
import scipy.io as scio
import numpy as np
import logging  
from datetime import datetime
from math import pi

GPU_ID = 3

# set log
from utils.common import init_log
OUTPUT_DIR = os.path.join('..', 'output', datetime.now().strftime('%Y%m%d-%H:%M'))
logging = init_log(OUTPUT_DIR)

# load model settings
logging.info('Train Test Settings: VGG16_LocNet_40_40_360')
from VGG16_LocNet_40_40_360 import \
    image_batch_size, pretrained_model, solver_file, \
    gen_pos_neg_bbox, regions_loc2who

# load proposals from faster rcnn results
from utils.minutiae_mse import load_pre_boxes
FASTER_PRO = os.path.join('.', 'nist27_4205_preboxes.mat')
pre_boxes = load_pre_boxes(FASTER_PRO)

sys.path.append(os.path.join('..', 'external', 'caffe_LocNet', 'python'))
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
def get_info_from_solver(solver_file=solver_file):
    solver_param = caffe_pb2.SolverParameter()
    with open(solver_file, 'rt') as f:
        pb2.text_format.Merge(f.read(), solver_param)
    return solver_param

def reshape_as_input(net, data_dict):
    for key in data_dict.keys():
        net.blobs[key].reshape(*data_dict[key].shape)
        net.blobs[key].data[...] = data_dict[key]
    net.reshape()
    return

from utils.common import image_preprocess
def data_prepare(solver, train_set, train_sample_rate, rand_shift=False):
    # random pick batch_size images from dataset
    set_pick = np.random.choice(len(train_set), 1, p=train_sample_rate)[0]
    pic_pick = np.random.choice(len(train_set[set_pick]), \
        image_batch_size, replace=False)
    # load image
    data_dict = {'data':[], 'regions_loc':[], 'label':[], 'loss_weight':[]}
    for i, idx in enumerate(pic_pick):
        pic_key = train_set[set_pick].keys()[idx]
        # prepare image data
        I = cv2.imread(pic_key)
        r_l, l, l_w = gen_pos_neg_bbox(
            pic_key, train_set[set_pick][pic_key], I.shape, i, False, rand_shift)
        if len(r_l) == 0: continue
        I = image_preprocess(I)
        data_dict['data'].append(I)
        data_dict['regions_loc'].extend(r_l)
        data_dict['label'].extend(l)
        data_dict['loss_weight'].extend(l_w)
    # prepare batch
    for key in data_dict.keys():
        data_dict[key] = np.asarray(data_dict[key])
    reshape_as_input(solver.net, data_dict)

def model_prepare(gpu_id=GPU_ID):
    logging.info('Prepare Model ...')
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    solver = caffe.get_solver(solver_file)
    solver.net.copy_from(pretrained_model)
    logging.info('Done.')
    return solver

from utils.minutiae_mse import evaluation
def model_test(solver, data_set, rand_shift=False):
    loss, pred_res, gene_res  = 0, {}, {}
    for i, pic_key in enumerate(data_set.keys()):
        # prepare image data
        I = cv2.imread(pic_key)
        r_l, l, l_w = gen_pos_neg_bbox( \
            pic_key, data_set[pic_key], I.shape, 0, True, rand_shift)
        if len(r_l) == 0: continue
        I = image_preprocess(I)
        data_dict = {'data':np.asarray([I]), \
            'regions_loc':np.asarray(r_l), \
            'label':np.asarray(l), \
            'loss_weight':np.asarray(l_w)}
        reshape_as_input(solver.net, data_dict)
        solver.net.forward()

        loss_temp = solver.net.blobs['loss_loc'].data
        loss += loss_temp
        pred_res[pic_key] = regions_loc2who( \
            solver.net.blobs['preds_loc'].data, data_dict['regions_loc'])
        gene_res[pic_key] = regions_loc2who( \
            data_dict['label'], data_dict['regions_loc'])    
        # ----------------------------------------------------------------------        
        # 暂且让pred的角度和gene的一致
        for i in xrange(len(pred_res[pic_key])):
            pred_res[pic_key][i][2] = gene_res[pic_key][i][2]
        # ----------------------------------------------------------------------

    logging.info('Average loss: %f'%(loss / len(data_set)))        
    logging.info('Predicte: Rec Pre_u Pre Loc Ori: %.3f, %.3f, %.3f, %.3f, %.3f' \
        %(evaluation(pred_res, data_set)))
    logging.info('Generate: Rec Pre_u Pre Loc Ori: %.3f, %.3f, %.3f, %.3f, %.3f' \
        %(evaluation(gene_res, data_set)))

from utils.common import moving_average
from load_data import train_valid_test_split
def model_train():
    train_set, valid_set, test_set = \
        train_valid_test_split(logging)
    rand_shift = True 
    train_sample_rate = (0, 1, 0)
    logging.info('Train: Nist27 0, 4205 1, Fvc2002 db2a 0')  
    # start validation results
    logging.info('Validation Init: Nist27')
    logging.info('Predicte: Rec Pre_u Pre Loc Ori: %.3f, %.3f, %.3f, %.3f, %.3f' \
        %(evaluation(pre_boxes, valid_set[0])))
    logging.info('Validation Init: 4205')
    logging.info('Predicte: Rec Pre_u Pre Loc Ori: %.3f, %.3f, %.3f, %.3f, %.3f' \
        %(evaluation(pre_boxes, valid_set[1]))) 
    logging.info('Test Init: Nist27')
    logging.info('Predicte: Rec Pre_u Pre Loc Ori: %.3f, %.3f, %.3f, %.3f, %.3f' \
        %(evaluation(pre_boxes, test_set[0])))    

    params = get_info_from_solver(solver_file)
    solver, loss, loss_rec = model_prepare(), None, []
    for _iter in xrange(params.max_iter):
        # prepare training data here need to be multiprocessing in future
        a = time.time()
        data_prepare(solver, train_set, train_sample_rate, rand_shift)
        b = time.time()
        # train model here and get the loss
        solver.step(1)
        c = time.time()
        loss_rec.append(solver.net.blobs['loss_loc'].data.tolist())
        loss = moving_average(loss_rec[-1], loss)
        if solver.iter % params.display == 0:
            region_pool5_loc = sum(sum(sum(solver.net.blobs['region_pool5_loc'].data[0, :, :, :])))
            pool7_x = sum(sum(sum(solver.net.blobs['pool7_x'].data[0, :, :, :])))
            pre_w = solver.net.blobs['fg_preds_x'].data[0, :].argmax() - 19
            pre_h = solver.net.blobs['fg_preds_y'].data[0, :].argmax() - 19
            pre_o = solver.net.blobs['fg_preds_o'].data[0, :].argmax()                      
            logging.info('Iter: step %d loss: %.4f' %(solver.iter, loss))
            logging.info('r_p, p7_x, p_w, p_h, p_o: %.3f, %.3f, %d, %d, %d' \
                %(region_pool5_loc, pool7_x, pre_w, pre_h, pre_o))
            fid = open(os.path.join(OUTPUT_DIR, 'loss.log'), 'a')
            for i, l_r in enumerate(loss_rec):
                fid.write('%.6f\n'%(l_r))
            fid.close()
            loss_rec = []

        if solver.iter % (params.display * 10) == 0:
        # if solver.iter % 1 == 0:
            logging.info('Train: Nist27')
            model_test(solver, train_set[0], True)            
            logging.info('Validation: Nist27')
            model_test(solver, valid_set[0])
            logging.info('Validation: 4205')
            model_test(solver, valid_set[1])
            logging.info('Validation: FVC2002 DB2A')
            model_test(solver, valid_set[2], True)                        
            logging.info('Test: Nist27')
            model_test(solver, test_set[0])

        if solver.iter % (params.max_iter / 10) == 0:
            logging.info('Saving snapshothot {0} ...'.format(_iter))
            filename = params.snapshot_prefix + \
                '_iter_{0}'.format(solver.iter) + '.caffemodel'
            filename = os.path.join(OUTPUT_DIR, filename)
            solver.net.save(str(filename))
            logging.info('Done.')

        d = time.time()
        print 'data prepare: %.5f, train: %.5f, others: %.5f'%(b - a, c - b, d - c)

from utils.show_minutiae import show_mnt
def train():
    model_train()


if __name__ == '__main__':
    train()