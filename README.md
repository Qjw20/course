# 基于DETR和CRNN的车牌检测识别模型
后端项目在main分支下，前端项目在master分支下，首先打开后端工程文件detr-main
## 数据集
本实验使用中科大的CCPD 2019开源数据集，数据集下载链接是：https://github.com/detectRecog/CCPD

接下来介绍将ccpd数据集转换成coco格式的步骤，一共4步，每步都包括一个py文件，在终端执行每个py文件之前，先把对应的路径换成自己数据的路径和自己保存结果的路径
### ccpd数据转换成yolo格式
```
python ccpd2yolo.py
```
### 随机取10000张图像
```
python cp10000.py
```
### 划分训练集和测试集
```
python splt_train_val.py
```
### yolo数据转换成coco格式
```
python yolo2coco.py
```
## 准备一个虚拟环境
虚拟环境配置：
```
python 3.8
pytorch 1.10.0
cuda 11.3
```
## DETR
### 安装
```
# 克隆存储库
git clone https://github.com/facebookresearch/detr.git
```
```
# 安装PyTorch 1.5+和torchvision 0.6+
conda install -c pytorch pytorch torchvision
```
```
# 安装pycocotools（用于COCO评估）和scipy（用于训练）
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
### 训练
将main.py文件中的coco_path换成自己数据的路径，output_dir换成自己保存结果的路径
在终端执行命令
```
python main.py
```
等待训练过程结束，将在自己保存结果的路径下看到训练结果文件
### 测试
找到test_detr.py文件，找到下述代码，按照注释提示修改3处代码，第1处修改为自己加载模型的路径，第2、3处修改为待预测图片所在文件夹路径
```
    # 加载模型
    dfdetr = load_model('output/checkpoint.pth',main_args) # <--修改为自己加载模型的路径
 
    files = os.listdir("/root/autodl-tmp/data/CCPD2019/val2017/") # <--修改为待预测图片所在文件夹路径
    ...
    for file in files:
        img_path = os.path.join("/root/autodl-tmp/data/CCPD2019/val2017/", file) # <--修改为待预测图片所在文件夹路径
```
找到下述代码，把路径/root/autodl-tmp/data/CCPD2019/crops2017/images修改为自己保存测试结果图像的路径
```
    if save_crops:
        if not os.path.exists("/root/autodl-tmp/data/CCPD2019/crops2017/images"):
            os.makedirs('/root/autodl-tmp/data/CCPD2019/crops2017/images')
        save_one_box(boxes, im, file='/root/autodl-tmp/data/CCPD2019/crops2017/images/{}'.format(save_name), BGR=True)
```
找到下述代码，把路径/root/autodl-tmp/data/CCPD2019/crops2017/labels修改为自己保存测试结果标签的路径
```
    if save_txt:  # Write to file(txt)
        ...
        txt_path = '/root/autodl-tmp/data/CCPD2019/crops2017/labels/{}'.format(save_name).replace('jpg', 'txt')
        if not os.path.exists("/root/autodl-tmp/data/CCPD2019/crops2017/labels"):
            os.makedirs('/root/autodl-tmp/data/CCPD2019/crops2017/labels')
```
等待测试过程结束，将在自己的测试结果的保存路径下看到测试结果文件
## CRNN
### 安装
在终端执行以下命令，安装paddleocr
```
python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```
### 测试
找到test_ocr.py，将测试数据集的路径、结果保存路径换成自己的路径
```
    images_dir = "/root/autodl-tmp/data/CCPD2019/crops2017/images" # 车牌目标检测框的裁剪图像
    new_dir ="/root/autodl-tmp/data/CCPD2019/recognize2017"
```
在终端执行命令
```
python test_ocr.py
```
等待测试过程结束，将在自己的测试结果的保存路径下看到测试结果文件
## 后端启动
在终端执行命令行启动后端
```
python app.py
```
打开前端工程文件license-plate-detection
## 前端启动
在终端执行命令行启动前端
```
npm run serve
```
前端成功启动后，终端会显示一个网址http://localhost:8080/，通过这个网址即可访问本系统
