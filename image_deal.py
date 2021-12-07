import pandas as pd
import shutil
import os
from PIL import Image
from PIL import ImageEnhance
import cv2
import numpy as np
import glob
import random

#宫颈细胞还是眼底数据
img_content='some_cell'

#数组增强是否开启
data_gen=False

#先删除train下的所有文件
train_path='data/train'
def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)
#先删除train下的所有文件
del_files(train_path)
print("train 的图片已经清空成功，开始生成数据")


#--------------------------------数据增强-----------------------------------------
def flip(root_path,img_name):   #翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img

def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(random.randint(0,360)) #旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def randomColor(root_path, img_name): #随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.open(os.path.join(root_path, img_name))
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度


def contrastEnhancement(root_path, img_name):  # 对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted

def brightnessEnhancement(root_path,img_name):#亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened

def colorEnhancement(root_path,img_name):#颜色增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    image_colored = enh_col.enhance(color)
    return image_colored

#数据增强函数
#是否开启，当前编号，名称前缀，旧路径，新路径
def data_generator(data_gen,num,name_fir,old_path,old_name,new_path):
    if data_gen==True:#开启数组增强

        # 翻转图像
        saveImage = flip(old_path, old_name)
        newname = name_fir + '.' + str(num) + '.jpg'
        saveImage.save(os.path.join(new_path, newname))
        num=num+1

        # 旋转图像
        saveImage = rotation(old_path, old_name)
        newname = name_fir + '.' + str(num) + '.jpg'
        saveImage.save(os.path.join(new_path, newname))
        num=num+1

        # 亮度增强
        saveImage = brightnessEnhancement(old_path, old_name)
        newname = name_fir + '.' + str(num) + '.jpg'
        #之后再随机旋转
        saveImage=saveImage.rotate(random.randint(0,360)) #旋转角度
        saveImage.save(os.path.join(new_path, newname))
        num = num + 1

        # 随机颜色
        saveImage = randomColor(old_path, old_name)
        newname = name_fir + '.' + str(num) + '.jpg'
        # 之后再随机旋转
        saveImage = saveImage.rotate(random.randint(0, 360))  # 旋转角度
        saveImage.save(os.path.join(new_path, newname))
        num = num + 1

        # 颜色增强
        saveImage = colorEnhancement(old_path, old_name)
        newname = name_fir + '.' + str(num) + '.jpg'
        # 之后再随机旋转
        saveImage = saveImage.rotate(random.randint(0, 360))  # 旋转角度
        saveImage.save(os.path.join(new_path, newname))
        num = num + 1

        # 对比度增强
        saveImage = contrastEnhancement(old_path, old_name)
        newname = name_fir + '.' + str(num) + '.jpg'
        # 之后再随机旋转
        saveImage = saveImage.rotate(random.randint(0, 360))  # 旋转角度
        saveImage.save(os.path.join(new_path, newname))
        num = num + 1

        return num

    if data_gen==False:
        return num


#----------------------------------开始处理------------------------------------------





csv_data_normal = pd.read_csv("normal.csv", usecols=['CLINICNO'])
origin_data_path = 'data/origin_data'
new_data_path = 'data/train'
# #处理origin_bad_data的图片名称
origin_bad_data_path='data/origin_bad_data'

# 正常的csv
normal_img_folder_second = []
# 读取cvs,并添加后缀.jpg
for index in range(len(csv_data_normal)):
    folder_name = csv_data_normal['CLINICNO'][index]
    normal_img_folder_second.append(folder_name)

all_img_folder = []
for img_folder_first in os.listdir(origin_data_path):
    img_folder_first_name = origin_data_path + '/' + img_folder_first
    for img_folder_second in os.listdir(img_folder_first_name):
        all_img_folder.append(int(img_folder_second))  # 转成整型

all_img_set = set(all_img_folder)

normal_set = set(normal_img_folder_second)
# 得到二级文件夹的交集

# 正常图片的二级文件夹
all_normal = all_img_set & normal_set
all_normal = list(all_normal)
print(all_normal)
print("正常图片的二级文件夹又：" + str(len(all_normal)))

# 所有正常的二级文件夹路径
all_normal_path_arr = []
for val in all_normal:
    all_normal_path_arr.append(origin_data_path + '/' + str(val)[:8] + '/' + str(val))
print(all_normal_path_arr)
print("正常的图片二级路径个数：" + str(len(all_normal_path_arr)))

# 读取正常图片路径，然后修改名字并且copy到train 中 命名规则 normal.nid.jpg
tid = 0
nid=0
for path in all_normal_path_arr:
    if tid >= 273:  # 保持数据集数量一样
        break
    if os.path.isdir(path) == True:
        tid = tid + 1
        for img in os.listdir(path):
            newname = 'normal.' + str(nid) + '.jpg'
            shutil.copyfile(os.path.join(path, img), os.path.join(new_data_path, newname))

            nid=nid+1
            # 采用数据增强 返回结果是新的数量
            nid = data_generator(data_gen, nid, 'normal', path, img, train_path)

print(nid)



un_num=0
for index, imgname in enumerate(os.listdir(os.path.join(origin_bad_data_path))):
        newname = 'unnormal.' + str(un_num) + '.jpg'
        shutil.copyfile(os.path.join(origin_bad_data_path, imgname), os.path.join(train_path, newname))
        un_num = un_num + 1

        # 采用数据增强 返回结果是新的数量
        un_num = data_generator(data_gen, un_num, 'unnormal', origin_bad_data_path, imgname, train_path)

