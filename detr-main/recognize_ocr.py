from paddleocr import PaddleOCR, draw_ocr
import os
import numpy as np
import cv2


# 在字符串指定位置插入字符
# str_origin：源字符串  pos：插入位置  str_add：待插入的字符串
def str_insert(str_origin, pos, str_add):
    str_list = list(str_origin)    # 字符串转list
    str_list.insert(pos, str_add)  # 在指定位置插入字符串
    str_out = ''.join(str_list)    # 空字符连接
    return  str_out

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    """
    使用opencv在原图im上画一个bounding box
    :params x: 预测得到的bounding box  [x1 y1 x2 y2]
    :params im: 原图 要将bounding box画在这个图上  array
    :params color: bounding box线的颜色
    :params labels: 标签上的框框信息  类别 + score
    :params line_thickness: bounding box的线宽
    """
    # check im内存是否连续
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    # tl = 框框的线宽  要么等于line_thickness要么根据原图im长宽信息自适应生成一个
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    # c1 = (x1, y1) = 矩形框的左上角   c2 = (x2, y2) = 矩形框的右下角
    c1, c2 = (int(float(x[0])), int(float(x[1]))), (int(float(x[2])), int(float(x[3])))
    # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle: 在im上画出框框   c1: start_point(x1, y1)  c2: end_point(x2, y2)
    # 这里的c1+c2可以是左上角+右下角  也可以是左下角+右上角都可以
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # 如果label不为空还要在框框上面显示标签label + score
    if label:
        tf = max(tl - 1, 1)  # label字体的线宽 font thickness
        # cv2.getTextSize: 根据输入的label信息计算文本字符串的宽度和高度
        # 0: 文字字体类型  fontScale: 字体缩放系数  thickness: 字体笔画线宽
        # 返回retval 字体的宽高 (width, height), baseLine 相对于最底端文本的 y 坐标
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # 同上面一样是个画框的步骤  但是线宽thickness=-1表示整个矩形都填充color颜色
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText: 在图片上写文本 这里是在上面这个矩形框里写label + score文本
        # (c1[0], c1[1] - 2)文本左下角坐标  0: 文字样式  fontScale: 字体缩放系数
        # [225, 255, 255]: 文字颜色  thickness: tf字体笔画线宽     lineType: 线样式
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    ocr = PaddleOCR(use_angle_cls=True, lang='ch',rec_algorithm='CRNN')

    # 测试pipline
    # images_dir = "/root/autodl-tmp/data/CCPD2019/ccpd10000/detect/exp/crops/licence" # 车牌目标检测框的裁剪图像
    # new_dir ="/root/autodl-tmp/data/CCPD2019/ccpd10000/recognition"
    images_dir = "results/detect/crops" # 车牌目标检测框的裁剪图像
    new_dir ="results/recognize"

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0

    n = len(os.listdir(images_dir))

    for i in range(n):
        image_path = os.path.join(images_dir, os.listdir(images_dir)[i]) 
        result = ocr.ocr(image_path, cls=True, det=False) 

        # 输出识别结果
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                print(line)

        from PIL import Image

        # original_image_path = "/root/autodl-tmp/data/CCPD2019/ccpd10000/images/" + os.listdir(images_dir)[i]
        original_image_path = "upload/" + os.listdir(images_dir)[i]
        txt_name = os.listdir(images_dir)[i][:-4] + ".txt"
        # box_path = "/root/autodl-tmp/data/CCPD2019/ccpd10000/detect/exp/labels/" + txt_name
        box_path = "results/detect/labels/" + txt_name
        # print(box_path)
        # read input file path
        boxes_line = []
        if os.path.isfile(box_path):
            with open(box_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                boxes_line.append(line.split("\n")[0])
        elif os.path.isdir(box_path):
            boxes_line = os.listdir(box_path)
        else:
            # raise Exception("the input file is file or dir")
            continue
        print(boxes_line)
        # 对boxes[0][2:]进行分割
        boxes = list()
        tmp = boxes_line[0][2:]
        lx,ly,rx,ry = tmp.split(' ', 3)
        lxy = list()
        lxy.append(int(float(lx)))
        lxy.append(int(float(ly)))
        l2xy = list()
        l2xy.append(int(float(lx)))
        l2xy.append(int(float(ry)))
        rxy = list()
        rxy.append(int(float(rx)))
        rxy.append(int(float(ry)))
        r2xy = list()
        r2xy.append(int(float(rx)))
        r2xy.append(int(float(ly)))

        boxes.append(lxy)
        boxes.append(l2xy)
        boxes.append(rxy)
        boxes.append(r2xy)

        box = list()
        box.append(boxes)

        print(box)

        image = cv2.imread(original_image_path)
        txts = [line[0][0] for line in result]
        scores = [line[0][1] for line in result]
        # print(scores)
        # 画出预测框
        im_show = draw_ocr(image, box, txts, scores, font_path='fonts/simfang.ttf')
        im_show = Image.fromarray(im_show)


        # provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
        # alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
        #             'X', 'Y', 'Z', 'O']
        # ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
        #     'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
        # filename = os.listdir(images_dir)[i]
        # result = ""
        # _, _, box, points, plate, brightness, blurriness = filename.split('-')
        # list_plate = plate.split('_')  # 读取车牌
        # result += provinces[int(list_plate[0])]
        # result += alphabets[int(list_plate[1])]
        # result += ads[int(list_plate[2])] + ads[int(list_plate[3])] + ads[int(list_plate[4])] + ads[int(list_plate[5])] + ads[int(list_plate[6])]

        # # 计数评估
        lic = os.listdir(images_dir)[i][:-4]
        # gt = str_insert(result, 2, '·')
        # print(gt)
        # for j, label in enumerate(txts):  # 统计准确率
        #     if len(label) != len(gt): # 去除.jpg后缀
        #         Tn_1 += 1  # 错误+1
        #         continue
        #     if (np.asarray(gt) == np.asarray(label)).all():
        #         Tp += 1  # 完全正确+1
        #     else:
        #         Tn_2 += 1
        
        result_img_dir = os.path.join(new_dir, "images")
        result_img_path = os.path.join(result_img_dir, lic) 
        # 测试piplines保存识别图像
        im_show.save(result_img_path + '.jpg')

        result_label_dir = os.path.join(new_dir, "labels")
        result_label_path = os.path.join(result_label_dir, lic) 
        with open(result_label_path + '.txt', 'a') as f:        
            f.write(str(result[0][0]))  # label format
    # Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)       
    # print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    # accuracy_path = os.path.join(new_dir, "accuracy") 

    # with open(accuracy_path + '.txt', 'a') as f:
    #     f.write(str("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2))))  # label format