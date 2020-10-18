# -*- coding: utf-8 -*-
# !@time: 2020/10/16 下午4:52
# !@author: superMC @email: 18758266469@163.com
# !@fileName: web_flask.py
from flask import Flask, render_template, Response
import cv2

from config import opt
from demo import Demo
from self_utils.models import init_models
from self_utils.utils import get_data

app = Flask(__name__)


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    is_video = True
    input_video = 'data/data1.avi'
    output_video = 'data/output.avi'
    dst_txt = 'data/output.txt'
    demo = Demo(input_video, output_video, dst_txt, models, is_video=is_video, database_labels=database_labels,
                database_features=database_features)
    return Response(demo.run(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    global models, database_labels, database_features
    models = init_models()
    database_labels, database_features = get_data(csv_path=opt.face_data_csv)
    app.run(host='10.20.216.1', debug=False, port=5000, use_reloader=False)
