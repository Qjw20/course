import shutil
import cv2
import os


# 1
def txt_translate(path, txt_path):
    for filename in os.listdir(path):
        print(filename)
        if not "-" in filename: #对于np等无标签的图片，过滤
            continue
        subname = filename.split("-", 3)[2]  # 第一次分割，以减号'-'做分割,提取车牌两角坐标
        extension = filename.split(".", 1)[1] #判断车牌是否为图片
        if not extension == 'jpg':
            continue
        lt, rb = subname.split("_", 1)  # 第二次分割，以下划线'_'做分割
        lx, ly = lt.split("&", 1) #左上角坐标
        rx, ry = rb.split("&", 1) # 右下角坐标
        width = int(rx) - int(lx) #车牌宽度
        height = int(ry) - int(ly)  # bounding box的宽和高
        cx = float(lx) + width / 2
        cy = float(ly) + height / 2  # bounding box中心点

        img = cv2.imread(os.path.join(path , filename))
        if img is None:  # 自动删除失效图片（下载过程有的图片会存在无法读取的情况）
            os.remove(os.path.join(path, filename))
            continue
        width = width / img.shape[1]
        height = height / img.shape[0]
        cx = cx / img.shape[1]
        cy = cy / img.shape[0]

        txtname = filename.split(".", 1)[0] +".txt"
        txtfile = os.path.join(txt_path, txtname)
        # 默认车牌为1类，标签为0
        with open(txtfile, "w") as f:
            f.write(str(0) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height))



if __name__ == '__main__':
    # 修改此处地址
    images_pre_dir = "/root/autodl-tmp/data/CCPD2019" # 换成放入图片的八万多张数据

    sub_name = ['ccpd_base', 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_rotate', 'ccpd_tilt', 'ccpd_weather']

    for name in sub_name:
        print(name)
        imaged_dir = os.path.join(images_pre_dir, name)
        txt_dir = os.path.join(images_pre_dir, name + "_labels")

        if not os.path.exists(txt_dir):
            os.mkdir(txt_dir)
    
        txt_translate(imaged_dir, txt_dir)
