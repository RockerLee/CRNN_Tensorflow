import numpy as np
import pandas as pd
import os
import utils
import config
import time
import cv2
import cv2 as cv
from sklearn.model_selection import train_test_split
from scipy.misc import imread, imresize, imsave

def denoise_file(filepath, outpath):
    t = time.time()
    start_ms = (int(round(t * 1000)))
    
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # increase contrast
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    cl1 = clahe.apply(gray)
    #blur = cv.medianBlur(cl1,5)
    #resize
    h = cl1.shape[0]
    w = cl1.shape[1]
    ratio = float(32) / float(h)
    dst_w = int(w * ratio)
    resized = cv2.resize(cl1, (dst_w, 32))#width=dst_w, height = 32
    
    cv.imwrite(outpath, resized)
    
    t = time.time()
    end_ms = (int(round(t * 1000)))
    duration = end_ms - start_ms
#     print('time elapse(ms):', duration) 

def preprocess_imgs(root_dir = '', start_index = 0):
    data = pd.read_csv(root_dir + 'train.csv')
    os.makedirs('preprocess', exist_ok=True)
    out_dir = root_dir + 'preprocess/'
    print('data loaded, format:')
    print(data.head(n=3))
    count = 0
    for filename, label in zip(data['filename'], data['label']):
        count += 1
        if count < start_index:
            continue
        file_path = root_dir + filename
        name = filename[filename.index('/')+1:]
        out_path = out_dir + name
        denoise_file(file_path, out_path)
        if count % 1000 == 0:
            print('process ' + file_path + ' to ' + out_path + ', finished, count = ' + str(count))
        

# label 'abc' -> [0, 1, 2]
def label_to_array(label):
    try:
        return [config.CHAR_VECTOR.index(x) for x in label]
    except Exception as ex:
        print(label)
        raise ex

def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), [i for i in range(len(seq))]))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1]+1], dtype=np.int64)
    # print('sparse_tuple_from, indices:', indices.shape, indices)
    # print('sparse_tuple_from, values:', values.shape, values)
    # print('sparse_tuple_from, shape:', shape)
    # print('sparse_tuple_from, indices.max(0):', indices.max(0))

    return indices, values, shape

def load_batches_internal(data):
    batches = []
    length = len(data)
    batch_num = int(np.ceil(length / config.BATCH_SIZE))
    print('load_batches_internal, BATCH_SIZE:', config.BATCH_SIZE, 'batch_num:', batch_num)
    for i in range(batch_num):
        # in a batch
        # if i >= 1:
        #     break
        # print('batch:', i, '--------------------------------------')
        start = i * config.BATCH_SIZE
        filename_list = []
        label_list = []
        if start + config.BATCH_SIZE < length:
            filename_list = data['filename'][start:start+config.BATCH_SIZE]
            label_list = data['label'][start:start+config.BATCH_SIZE]
        else:
            filename_list = data['filename'][start:]
            label_list = data['label'][start:]
        
        img_batch = []
        label_batch = []
        array_batch = []
        for filename, label in zip(filename_list, label_list):
            filename = filename.replace('train', 'preprocess') # train/0.jpg -> preprocess/0.jpg
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 32 * 150
            img_batch.append(gray)
            label_batch.append(label)
            array_batch.append(label_to_array(label))
        
        # print('filename_list', filename_list)
        # print('label_list', label_list)
        # print('img_batch', img_batch)
        # print('label_batch', label_batch)
        # print('array_batch', array_batch)
        sparse_targets = sparse_tuple_from(array_batch)
        print('batch finish:', i)
        img_batch = np.asarray(img_batch)
        h = img_batch.shape[1]
        w = img_batch.shape[2]
        batches.append((np.asarray(label_batch), sparse_targets, np.reshape(img_batch, (config.BATCH_SIZE,h,w,1))))
    return batches

# shape: [batch_num], item: ([label...], [label_to_array...], [batch_size*32*width])
def load_train_batches(root_dir = ''):
    train_batches = []
    data = pd.read_csv(root_dir + 'train.csv')
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=config.RANDOM_STATE)
    print('train data size:', len(data_train))
    print('test data size:', len(data_test))

    print('loading data... ')
    start_time = time.time()
    train_batches = load_batches_internal(data_train)
    print('load finished, elapsed:', (time.time() - start_time), '=============================================')
    return train_batches

def load_test_batches(root_dir = ''):
    test_batches = []
    data = pd.read_csv(root_dir + 'train.csv')
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=config.RANDOM_STATE)

    print('loading data... ')
    start_time = time.time()
    test_batches = load_batches_internal(data_test)
    print('load finished, elapsed:', (time.time() - start_time))
    return test_batches

def ground_truth_to_word(ground_truth):
    try:
        return ''.join([config.CHAR_VECTOR[i] for i in ground_truth if i != -1])
    except Exception as ex:
        print(ground_truth)
        print(ex)