import os 
import shutil
import random

# 2
images_pre_dir = "/root/autodl-tmp/data/CCPD2019" # 换成放入图片的八万多张数据

sub_name = ['ccpd_base', 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_rotate', 'ccpd_tilt', 'ccpd_weather']

results_dir = "/root/autodl-tmp/data/CCPD2019/images" # 换成放入图片的八万多张数据

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

for name in sub_name:
    print(name)
    images_dir = os.path.join(images_pre_dir, name + "_images")
    new_dir = os.path.join(images_pre_dir, name + "_1250/images")

    if not os.path.exists(new_dir):
        if not os.path.exists(os.path.join(images_pre_dir, name + "_1250")):
            os.mkdir(os.path.join(images_pre_dir, name + "_1250"))
        os.mkdir(new_dir)

    n = len(os.listdir(images_dir))
    random.shuffle(os.listdir(images_dir))
    for i in range(1250):
        image_path = os.path.join(images_dir, os.listdir(images_dir)[i]) 
        shutil.copy(image_path,os.path.join(results_dir, os.listdir(images_dir)[i]))
        label_name = os.listdir(images_dir)[i].replace(".jpg",".txt")
        label_path =image_path.replace(".jpg",".txt").replace("images","labels")
        target_path = os.path.join(results_dir.replace("images","labels"))
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        shutil.copy(label_path,target_path + "/" + label_name)
        print(i)