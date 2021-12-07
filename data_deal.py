import cv2 as cv
import os
import numpy as np

import random
import pickle

import time

start_time = time.time()

data_dir = './data'
batch_save_path = './batch_files'
train_dir='train'+'/'

fir='normal'
sec='unnormal'
img_size=100 #图片大小

# 输入总数量
all_num=3871

# 训练测试比率
radio=0.8

# 最小数据得是100的倍数，求余，然后减去余数获得完全的整数，这样好按照radio的比率划分训练集和测试集以及他们的batch大小
# 如果划分不好，那么再训练集和测试集的数组部分就会出现取到空数组的问题，导致模型训练的np.vstack 垂直排列和np.hstack 水平排列会出现维度的问题
yu=all_num%100

all_num_except_yu=all_num-yu

# 按照radio的比率划分训练集和测试集，测试集是打算从剩下的到最后，所以不用乘 1-radio ,用减法就行
train_data_num=all_num_except_yu*radio

test_data_num=all_num-train_data_num

# batch文件一个存n个图片，所以训练的batch总共是 batchs_num=train_data_num/n 个
every_n=76
batchs_num=40

assert every_n*batchs_num==train_data_num
print("训练集batch共"+str(batchs_num)+"个，每个batch含有"+str(every_n)+"个图片，请将batchs_num的个数" +str(batchs_num)+ "填到tensor.py的头部")

# 创建batch文件存储的文件夹
os.makedirs(batch_save_path, exist_ok=True)

# 图片统一大小：100 * 100
# 训练集 20000：100个batch文件，每个文件200张图片
# 验证集 5000： 一个测试文件，测试时 50张 x 100 批次

# 进入图片数据的目录，读取图片信息
all_data_files = os.listdir(os.path.join(data_dir, train_dir))

# 打算数据的顺序
random.shuffle(all_data_files)

#从0到train_data_num
all_train_files = all_data_files[:int(train_data_num)]

#从train_data_num到最后
all_test_files = all_data_files[int(train_data_num):]

train_data = []
train_label = []
train_filenames = []

test_data = []
test_label = []
test_filenames = []

# 训练集
it=0
for each in all_train_files:
    img = cv.imread(os.path.join(data_dir,train_dir,each),1)
    resized_img = cv.resize(img, (img_size,img_size))

    img_data = np.array(resized_img)
    train_data.append(img_data)
    if fir ==each.split('.')[0]:
        #print('normal in list')
        train_label.append(0)
    elif sec ==each.split('.')[0]:
        #print('un normal in list')
        train_label.append(1)
    else:
        raise Exception('%s is wrong train file'%(each))
    train_filenames.append(each)
    it=it+1
    if it %100==0:
        print("训练集，第"+str(it)+"步已经完成")

# 测试集
ite=0
for each in all_test_files:
    img = cv.imread(os.path.join(data_dir,train_dir,each), 1)
    resized_img = cv.resize(img, (img_size,img_size))

    img_data = np.array(resized_img)
    test_data.append(img_data)
    if fir ==each.split('.')[0]:
        #print('normal in list')
        test_label.append(0)
    elif sec ==each.split('.')[0]:
        #print('un normal in list')
        test_label.append(1)
    else:
        raise Exception('%s is wrong test file'%(each))
    test_filenames.append(each)
    ite=ite+1
    if ite%100==0:
        print("测试集，第" + str(ite) + "已经完成")

print(len(train_data), len(test_data))

# 制作batchs_num个batch文件，每个batch有every_n个图片和其他数据
start = 0
end = every_n
for num in range(1, batchs_num+1):
    batch_data = train_data[start: end]
    batch_label = train_label[start: end]
    batch_filenames = train_filenames[start: end]
    batch_name = 'training batch {} of 15'.format(num)
    print("当前lable")
    print(batch_label)
    all_data = {
        'data':batch_data,
        'label':batch_label,
        'filenames':batch_filenames,
        'name':batch_name
    }

    with open(os.path.join(batch_save_path, 'train_batch_{}'.format(num)), 'wb') as f:
        pickle.dump(all_data, f)

    # 数组索引后移
    start += every_n
    end += every_n

    print("训练batch，第" + str(num) + "已经完成")

# 制作测试文件
all_test_data = {
    'data':test_data,
    'label':test_label,
    'filenames':test_filenames,
    'name':'test batch 1 of 1'
}

with open(os.path.join(batch_save_path, 'test_batch'), 'wb') as f:
    pickle.dump(all_test_data, f)
print("测试batch，已经完成")


end_time = time.time()
print('制作结束, 用时{}秒'.format(end_time - start_time))