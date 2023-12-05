import cv2
from PIL import Image
import numpy as np
import os
import time
 
import torch
from torch import nn
import torchvision.transforms as T
from main import get_args_parser as get_main_args_parser
from models import build_model
 

import re           # 用来匹配字符串（动态、模糊）的模块
import glob
from pathlib import Path


torch.set_grad_enabled(False)
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] 当前使用{}做推断".format(device))
 
# 图像数据处理
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
# plot box by opencv
def plot_result(pil_img, prob, boxes, save_name=None, imshow=False, imwrite=True, save_crops=True, save_txt=True):
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    LABEL =['license']
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes):
        cl = p.argmax()
        label_text = '{}: {}%'.format(LABEL[cl], round(p[cl] * 100, 2))
 
        cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
        cv2.putText(opencvImage, label_text, (int(xmin) + 10, int(ymin) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)
    
    if imshow:
        cv2.imshow('detect', opencvImage)
        cv2.waitKey(0)
 
    if imwrite:
        if not os.path.exists("results/detect/imgs"):
            os.makedirs('results/detect/imgs')
        cv2.imwrite('results/detect/imgs/{}'.format(save_name), opencvImage)


    im = cv2.imread('/root/autodl-tmp/data/CCPD2019/val2017/{}'.format(save_name))

    if save_crops:
        if not os.path.exists("/root/autodl-tmp/data/CCPD2019/crops2017/images"):
            os.makedirs('/root/autodl-tmp/data/CCPD2019/crops2017/images')
        save_one_box(boxes, im, file='/root/autodl-tmp/data/CCPD2019/crops2017/images/{}'.format(save_name), BGR=True)

    if save_txt:  # Write to file(txt)
        # 将xyxy(左上角 + 右下角)格式转换为xywh(中心的 + 宽高)格式 并除以gn(whwh)做归一化 转为list再保存
        line = '0' + ' ' + str(boxes[0][0]) + ' ' + str(boxes[0][1]) + ' ' + str(boxes[0][2]) + ' ' + str(boxes[0][3])
        print("line:", line)
        txt_path = '/root/autodl-tmp/data/CCPD2019/crops2017/labels/{}'.format(save_name).replace('jpg', 'txt')
        if not os.path.exists("/root/autodl-tmp/data/CCPD2019/crops2017/labels"):
            os.makedirs('/root/autodl-tmp/data/CCPD2019/crops2017/labels')
        with open(txt_path, 'a') as f:
            f.write(line+ '\n')


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


# 将xywh转xyxy
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
 
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b
 
def load_model(model_path , args):
    model, _, _ = build_model(args)
    model.cuda()
    model.eval()
    state_dict = torch.load(model_path) # <-----------修改加载模型的路径
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("load model sucess")
    return model
 
# 图像的推断
def detect(im, model, transform, prob_threshold=0.7):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
 
 
    # propagate through the model
    img = img.to(device)
    start = time.time()
    outputs = model(img)
   
    # keep only predictions with 0.7+ confidence
    print(outputs['pred_logits'].softmax(-1)[0, :, :-1])
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > prob_threshold
 
    probas = probas.cpu().detach().numpy()
    keep = keep.cpu().detach().numpy()
 
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    end = time.time()

    return probas[keep], bboxes_scaled, end - start


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """这是个用处特别广泛的函数 train.py、detect.py、test.py等都会用到
    递增路径 如 run/train/exp --> runs/train/exp{sep}0, runs/exp{sep}1 etc.
    :params path: window path   run/train/exp
    :params exist_ok: False
    :params sep: exp文件名的后缀  默认''
    :params mkdir: 是否在这里创建dir  False
    """
    path = Path(path)  # string/win路径 -> win路径
    # 如果该文件夹已经存在 则将路径run/train/exp修改为 runs/train/exp1
    if path.exists() and not exist_ok:
        # path.suffix 得到路径path的后缀  ''
        suffix = path.suffix
        # .with_suffix 将路径添加一个后缀 ''
        path = path.with_suffix('')
        # 模糊搜索和path\sep相似的路径, 存在一个list列表中 如['runs\\train\\exp', 'runs\\train\\exp1']
        # f开头表示在字符串内支持大括号内的python表达式
        dirs = glob.glob(f"{path}{sep}*")
        # r的作用是去除转义字符       path.stem: 没有后缀的文件名 exp
        # re 模糊查询模块  re.search: 查找dir中有字符串'exp/数字'的d   \d匹配数字
        # matches [None, <re.Match object; span=(11, 15), match='exp1'>]  可以看到返回span(匹配的位置) match(匹配的对象)
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        # i = [1]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        # 生成需要生成文件的exp后面的数字 n = max(i) + 1 = 2
        n = max(i) + 1 if i else 1  # increment number
        # 返回path runs/train/exp2
        path = Path(f"{path}{sep}{n}{suffix}")  # update path

    # path.suffix文件后缀   path.parent　路径的上级目录  runs/train/exp2
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:  # mkdir 默认False 先不创建dir
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path  # 返回runs/train/exp2


def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
# def save_one_box(xywh, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    """用在detect.py文件中  由opt的save-crop参数控制执不执行
    将预测到的目标从原图中扣出来 剪切好 并保存 会在runs/detect/expn下生成crops文件，将剪切的图片保存在里面
    Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    :params xyxy: 预测到的目标框信息 list 4个tensor x1 y1 x2 y2 左上角 + 右下角
    :params im: 原图片 需要裁剪的框从这个原图上裁剪  nparray  (1080, 810, 3)
    :params file: runs\detect\exp\crops\dog\bus.jpg
    :params gain: 1.02 xyxy缩放因子
    :params pad: xyxy pad一点点边界框 裁剪出来会更好看
    :params square: 是否需要将xyxy放缩成正方形
    :params BGR: 保存的图片是BGR还是RGB
    :params save: 是否要保存剪切的目标框
    """
    xyxy = torch.tensor(xyxy).view(-1, 4)  # list -> Tensor [1, 4] = [x1 y1 x2 y2]
    b = xyxy2xywh(xyxy)  # xyxy to xywh [1, 4] = [x y w h]
    # b = xywh
    if square:  # 一般不需要rectangle to square
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    # box wh * gain + pad  box*gain再加点pad 裁剪出来框更好看
    b[:, 2:] = b[:, 2:] * gain + pad
    # xyxy = xywh2xyxy(b).long()  # xywh -> xyxy
    xyxy = box_cxcywh_to_xyxy(b).long()
    # 将boxes的坐标(x1y1x2y2 左上角右下角)限定在图像的尺寸(img_shape hw)内
    clip_coords(xyxy, im.shape)
    # crop: 剪切的目标框hw
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        # 保存剪切的目标框
        cv2.imwrite(file, crop)

    return crop
 
if __name__ == "__main__":
    
    main_args = get_main_args_parser().parse_args()
    # 加载模型
    dfdetr = load_model('output/checkpoint.pth',main_args) # <--修改为自己加载模型的路径
 
    files = os.listdir("/root/autodl-tmp/data/CCPD2019/val2017/") # <--修改为待预测图片所在文件夹路径
 
    cn = 0
    waste=0
    for file in files:
        img_path = os.path.join("/root/autodl-tmp/data/CCPD2019/val2017/", file) # <--修改为待预测图片所在文件夹路径
        im = Image.open(img_path)

 
        scores, boxes, waste_time = detect(im, dfdetr, transform)
        plot_result(im, scores, boxes, save_name=file, imshow=False, imwrite=False, save_crops=True, save_txt=True)
        print("{} [INFO] {} time: {} done!!!".format(cn,file, waste_time))
 
        cn+=1
        waste+=waste_time
        waste_avg = waste/cn
        print(waste_avg)