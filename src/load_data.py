import os
import glob
import random as rand
TRAIN_TEST = os.path.join('.', 'config_trainingset_balance.mat')

def mnt_reader(file_name):
    f = open(file_name)
    ground_truth = []
    for i, line in enumerate(f):
        if i < 2 or len(line) == 0: continue
        try:
            w, h, o = [float(x) for x in line.split()]
            w, h = int(round(w)), int(round(h))
            ground_truth.append([w, h, o])
        except:
            pass
    return ground_truth

def load_nist27():
    image_base = '/media/ssd2/tangy/nist27/latent_p/'
    image_files = glob.glob(os.path.join(image_base, "*.bmp"))
    mnt_base = '/media/ssd2/tangy/nist27/latent_m_MNT/'
    data = {}
    for image in image_files:
        _, name = os.path.split(image)
        name, _ = os.path.splitext(name)
        minutiae = mnt_reader(mnt_base+name+'.mnt')
        if len(minutiae) > 0:
            data[image] = minutiae
    return data


def load_4205():
    image_base = '/media/ssd2/tangy/data_finger_pad/'
    image_files = glob.glob(os.path.join(image_base, "*.bmp"))
    mnt_base = '/media/ssd2/tangy/data_finger_pad/'
    data = {}
    for image in image_files:
        _, name = os.path.split(image)
        name, _ = os.path.splitext(name)
        minutiae = mnt_reader(mnt_base+name+'.mnt')
        if len(minutiae) > 0:
            data[image] = minutiae
    return data

def load_fvc2002_db2a():
    image_base = '/media/ssd2/tangy/FVC/FVC2002/DB2_A/'
    image_files = glob.glob(os.path.join(image_base, "*.bmp"))
    mnt_base = '/media/ssd2/tangy/FVC/FVC2002/DB2_A_mnt/'
    data = {}
    for image in image_files:
        _, name = os.path.split(image)
        name, _ = os.path.splitext(name)
        minutiae = mnt_reader(mnt_base+name+'.mnt')
        if len(minutiae) > 0:
            data[image] = minutiae
    return data

def train_valid_split(dataset, split=0.9):
    train, valid = {}, {}
    for key in dataset.keys():
        if rand.random() > split:
            valid[key] = dataset[key]
        else:
            train[key] = dataset[key]
    return train, valid 

import scipy.io as scio
def train_valid_test_split(logging):
    logging.info('Prepare Dataset ...')
    nist27 = load_nist27()  
    m4205 = load_4205()
    fvc2002_db2a = load_fvc2002_db2a()
  
    nist27_names  = scio.loadmat(TRAIN_TEST)['Names'][:, 0]
    nist27_test_l = scio.loadmat(TRAIN_TEST)['idx_test']
    nist27_test_l = [nist27_names[i - 1][0][0] for i in nist27_test_l]
    nist27_train_valid, nist27_test = {}, {}
    for key in nist27.keys():
        _, basename = os.path.split(key)
        if basename[:-4] in nist27_test_l:
            nist27_test[key] = nist27[key]
        else:
            nist27_train_valid[key] = nist27[key]

    nist27_train, nist27_valid = train_valid_split(nist27_train_valid)
    m4205_train, m4205_valid = train_valid_split(m4205)
    fvc2002_db2a_train, fvc2002_db2a_valid = train_valid_split(fvc2002_db2a)

    train_set = [nist27_train, m4205_train, fvc2002_db2a_train]
    valid_set = [nist27_valid, m4205_valid, fvc2002_db2a_valid]
    test_set = [nist27_test]
    logging.info('Train Test Split: config_trainingset_balance.mat.')
    logging.info('Train:')
    logging.info('Nist27: %d m4205: %d fvc2002 db2a: %d' \
        %(len(nist27_train), len(m4205_train), len(fvc2002_db2a_train)))
    logging.info('Valid:')
    logging.info('Nist27: %d m4205: %d fvc2002 db2a: %d' \
        %(len(nist27_valid), len(m4205_valid), len(fvc2002_db2a_valid)))  
    logging.info('Test:')
    logging.info('Nist27: %d' %(len(nist27_test)))      
    logging.info('Done.')
    train_sample_rate = (0.5, 0.5, 0.0)
    logging.info('Train sample rage: Nist27 50%, 4205 25%, Fvc2002 db2a 25%')    
    return train_set, valid_set, test_set, train_sample_rate