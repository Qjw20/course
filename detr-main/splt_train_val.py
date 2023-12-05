# -*- coding: utf-8 -*-
import os
from os import listdir, getcwd
from os.path import join
import random
import shutil


# 3
# images_pre_dir = "/root/autodl-tmp/data/CCPD2019" # 换成放入图片的八万多张数据

# sub_name = ['ccpd_base', 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_rotate', 'ccpd_tilt', 'ccpd_weather']

# for name in sub_name:
#     wd = os.path.join(images_pre_dir, name + "_1250")

wd = "/root/autodl-tmp/data/CCPD2019" # 换成放入图片的八万多张数据

detection = False 

if detection:
    annotation_dir = os.path.join(wd, "labels/")
    if not os.path.isdir(annotation_dir):
        raise Exception("label dictory not found")
    image_dir = os.path.join(wd, "images/")
    if not os.path.isdir(image_dir):
        raise Exception("image dictory not found")

    train_file = open(os.path.join(wd, "train.txt"), 'w')
    val_file = open(os.path.join(wd, "val.txt"), 'w')
    train_file.close()
    val_file.close()

    train_file = open(os.path.join(wd, "train.txt"), 'a')
    val_file = open(os.path.join(wd, "val.txt"), 'a')

    list = os.listdir(image_dir) # list image files
    probo = random.randint(1, 100)
    print (len(list))
    for i in range(0, len(list)):
        path = os.path.join(image_dir,list[i])
        print(path)
        if os.path.isfile(path):
            image_path = image_dir + list[i]
            voc_path = list[i]
            (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
            (voc_nameWithoutExtention, voc_extention) = os.path.splitext(os.path.basename(voc_path))
            annotation_name = nameWithoutExtention + '.txt'
            annotation_path = os.path.join(annotation_dir, annotation_name)
            print (annotation_path)
        probo = random.randint(1, 100)
        print("Probobility: %d" % probo)
        if(probo <= 80):
            if os.path.exists(annotation_path):
                train_file.write(image_path + '\n')
        else:
            if os.path.exists(annotation_path):
                val_file.write(image_path + '\n')
    train_file.close()
    val_file.close()
else:
    # train文件夹
    train_txt_path = os.path.join(wd, "train.txt")
    train_dir = os.path.join(wd, "train/")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if os.path.isfile(train_txt_path):
        with open(train_txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            image_path = line.split("\n")[0]
            img_name = image_path.split("images/")[1]
            print(img_name)
            print(image_path)
            val_path = os.path.join(train_dir,img_name)
            print(val_path)
            # 用shutil.copy按张复制
            shutil.copy(image_path, val_path)
    # val文件夹
    val_txt_path = os.path.join(wd, "val.txt")
    val_dir = os.path.join(wd, "val/")
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    if os.path.isfile(val_txt_path):
        with open(val_txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            image_path = line.split("\n")[0]
            img_name = image_path.split("images/")[1]
            print(img_name)
            print(image_path)
            val_path = os.path.join(val_dir,img_name)
            print(val_path)
            # 用shutil.copy按张复制
            shutil.copy(image_path, val_path)

