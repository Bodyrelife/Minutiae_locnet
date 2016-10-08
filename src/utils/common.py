#coding=utf-8 
import os
import shutil
import logging
import numpy as np
from math import pi, cos, sin
def r_int(num):
    return int(round(num))

def image_preprocess(I, mean=[0.48501959, 0.45795685, 0.40760392]):
    I = np.asarray(I, dtype=np.float32) / 255
    for c in xrange(I.shape[2]):
        I[:, :, c] -= mean[c]
    I = I.transpose((2, 0, 1))
    return I

def gen_rand_shift(max_shift, min_shift = 0):
    rand_shift = np.random.rand() * max_shift + min_shift
    rand_angle = np.random.rand() * 2 * pi
    rand_w_h = np.asarray([rand_shift * cos(rand_angle), \
        rand_shift * sin(rand_angle)])
    return [r_int(x) for x in rand_w_h]

def re_mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def init_log(output_dir):
    re_mkdir(output_dir)
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=os.path.join(output_dir, 'log.log'),
        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

def moving_average(new, old, expo=0.1):
    if old is None:
        return new
    else:
        return expo * new + (1-expo) * old
