from flask import Flask,request, make_response
import os
from flask_cors import CORS, cross_origin
import cv2
import base64
import subprocess

app = Flask(__name__)
CORS(app)

@cross_origin()
@app.route("/file_rec", methods=["POST", "GET"])
def file_receive():
    print(request.files)
    file = request.files.get("file")
    file_name = request.files.get("file").filename
    # print(file_name)
    print(file)
    if file is None:# 表示没有发送文件
        return {
            'message': "文件上传失败"
        }
    # file_name = file.filename.replace(" ","")
    print("获取上传文件的名称为[%s]\n" % file_name)
    file.save('upload/' + file_name)  # 保存文件

    return {
        'code': 200,
        'messsge': "文件上传成功",
        'fileName': file_name,
        'url': 'http://127.0.0.1:5000/upload/' + file_name
    }

@cross_origin()
@app.route("/detect_recognize", methods=["POST", "GET"])
def file_detect_recognize():
    # 读取文件夹下的图片
    # 调用模型进行detect
    # 保存带框的结果
    # 裁切detect结果图片
    # with open("detect_yolov5.py", "r") as f:
    #    code = f.read()
    # exec(code)
    subprocess.call(['python', 'detect_detr.py'])

    # 调用crnn进行recognize
    # with open("recognize_ocr.py", "r") as f:
    #     code = f.read()
    # exec(code)
    subprocess.call(['python', 'recognize_ocr.py'])

    # 返回recognize结果和带框的图像
    images_dir = "results/detect/imgs" # 车牌目标检测框的裁剪图像
    
    n = len(os.listdir(images_dir))
    url_plates = list()    
    for i in range(n):
        url_plate = list()
        # 调用函数获取图片的URL
        image_path = os.path.join(images_dir, os.listdir(images_dir)[i]) 
        image_url = get_image_url(image_path)
        # print(image_url)
        url_plate.append(image_url)
        # 获取车牌号
        lic = os.listdir(images_dir)[i][:-4]
        result_label_path = os.path.join("results/detect/labels", lic) 
        with open(result_label_path + ".txt", 'r') as f:
            lines = f.readlines()
            print(lines[0])
            url_plate.append(lines[0])
        url_plates.append(url_plate)
    resp = make_response(url_plates)
    print(resp)
    #设置response的headers对象
    # resp.headers['Content-Type'] = 'image/jpg'
    return resp


def get_image_url(image_path):
    # 假设图片存储在本地，可以使用以下代码获取图片的URL
    with open(image_path, 'rb') as file:
        image_data = file.read()
        image_url = 'data:image/jpeg;base64,' + base64.b64encode(image_data).decode('utf-8')
    return image_url

# web 服务器
if __name__ == '__main__':
    app.run()

