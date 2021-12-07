import numpy as np
import tensorflow as tf
import pickle
# 生成了TFRecords文件，接下来就可以使用队列（queue）读取数据了
def load_data(filename):
    '''从batch文件中读取图片信息'''
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
        return data['data'],data['label'],data['filenames']








print("-----------------------------------------------------")

for i in range(1,2):
    print(i)