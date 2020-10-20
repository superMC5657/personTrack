# -*- coding: utf-8 -*-
# !@time: 2020/10/16 下午4:52
# !@author: superMC @email: 18758266469@163.com
# !@fileName: web_flask.py
import os
import time

from flask import Flask, render_template, Response, request, send_from_directory, make_response
from werkzeug.utils import secure_filename

from config import opt
from demo import Demo
from self_utils.models import init_models
from self_utils.utils import get_data

app = Flask(__name__)
dir_path = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(dir_path, 'upload/')
app.config['OUTPUT_FOLDER'] = os.path.join(dir_path, 'output/')


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        global demo
        f = request.files['file']
        videoname = secure_filename(f.filename)
        rstp = request.form.get('remote')
        is_video = request.form.get('input_type') == 'video'
        if is_video:
            filename = videoname
            src_video = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(src_video)
        else:
            src_video = rstp
            filename = str(time.time()) + '.avi'
        dst_video = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        dst_txt = os.path.join(app.config['OUTPUT_FOLDER'], os.path.splitext(filename)[0] + '.txt')

        demo = Demo(src_video, dst_video, dst_txt, models, is_video=is_video, database_labels=database_labels,
                    database_features=database_features)

        return render_template('index.html', filename=filename, is_video=is_video,
                               download_list=[filename, os.path.splitext(filename)[0] + '.txt'])


@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    if request.method == 'GET':
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(demo.run(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    global models, database_labels, database_features
    models = init_models()
    database_labels, database_features = get_data(csv_path=opt.face_data_csv)
    app.run(host='10.20.216.1', debug=True, port=5000, use_reloader=True)
