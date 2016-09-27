#coding=utf-8 
import os
import shutil
import logging

def r_int(num):
    return int(round(num))

def wh2tlbr(wh, shape):
    # 注意看region pooling layer的代码
    w, h = wh
    tlbr = [max(w-19, 0), max(h-19, 0), \
        min(w+20, shape[1]), min(h+20, shape[0])]
    return tlbr

def tlbr2wh(tlbr):
    # 注意看region pooling layer的代码
    tlw, tlh, brw, brh = tlbr
    w = r_int((tlw + brw) / 2.)
    h = r_int((tlh + brh) / 2.)
    return (w, h)

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

def moving_average(new, old, expo=0.9):
    if old is None:
        return new
    else:
        return expo * new + (1-expo) * old