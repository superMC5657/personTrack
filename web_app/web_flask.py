# -*- coding: utf-8 -*-
# !@time: 2020/10/16 下午4:52
# !@author: superMC @email: 18758266469@163.com
# !@fileName: web_flask.py
from flask import Flask, render_template, Response
import cv2
from demo import Demo
from self_utils.models import init_models


class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture("data/data1.avi")

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        if success:
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        else:
            return None


app = Flask(__name__)


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    input_video = 'data/data1.avi'
    output_video = 'data/output.avi'
    dst_txt = 'data/output.txt'
    models = init_models()
    is_video = True
    demo = Demo(input_video, output_video, dst_txt, models, is_video=is_video)
    return Response(demo.run(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='10.20.216.1', debug=True, port=5000)
