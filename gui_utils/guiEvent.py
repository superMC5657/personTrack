# -*- coding: utf-8 -*-
# !@time: 2020/9/6 下午7:55
# !@author: superMC @email: 18758266469@163.com
# !@fileName: guiEvent.py
import os
import threading

import cv2
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog

from config import opt
from demo import Demo


class GuiEvent:
    def __init__(self, ui, mainWnd):
        self.ui = ui
        self.mainWnd = mainWnd

        # 默认视频源为相机
        self.ui.radioButtonVideo.setChecked(True)
        self.isVideo = True

        # 信号槽设置
        ui.open.clicked.connect(self.open)
        ui.close.clicked.connect(self.close)

        ui.radioButtonCam.clicked.connect(self.radioButtonCam)
        ui.radioButtonVideo.clicked.connect(self.radioButtonVideo)

        ui.find_video.clicked.connect(self.find_video)
        ui.find_output.clicked.connect(self.find_output)

        # 创建一个关闭事件并设为未触发
        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        self.ui.close.setEnabled(False)

    def callback_progress(self, percentage):
        self.ui.video_progressBar.setProperty("value", percentage * 100)

    def callback_video(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
        self.ui.DispalyLabel.setPixmap(QPixmap.fromImage(showImage))
        if self.stopEvent.is_set():
            # 关闭事件置为未触发，清空显示label
            self.stopEvent.clear()
            return True
        return False

        # self.ui.DispalyLabel.clear()

        # pix = QtGui.QPixmap.fromImage(showImage)
        # self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        # self.scene = QGraphicsScene()  # 创建场景
        # self.scene.addItem(self.item)
        # self.ui.graphicsView.setScene(self.scene)  # 将场景添加至视图

    def radioButtonCam(self):
        self.isVideo = False

    def radioButtonVideo(self):
        self.isVideo = True

    def find_video(self):
        self.fileName, self.fileType = QFileDialog.getOpenFileName(self.mainWnd, 'Choose file', '', '*')
        self.ui.input_video_path.setPlainText(self.fileName)

    def find_output(self):
        self.output_dir = QFileDialog.getExistingDirectory(self.mainWnd, 'Choose dir', '')
        self.ui.output_path.setPlainText(self.output_dir)
        self.output_video = os.path.join(self.output_dir, opt.default_output_name + '.avi')
        self.output_txt = os.path.join(self.output_dir, opt.default_output_name + '.txt')

    def open(self):
        self.ui.video_progressBar.setProperty("value", 0)
        self.ui.DispalyLabel.clear()
        self.stopEvent.clear()
        self.ui.close.setEnabled(True)
        self.ui.open.setEnabled(False)
        if self.isVideo:
            self.demo = Demo(self.fileName, self.output_video, self.output_txt, self.callback_progress,
                             self.callback_video,
                             self.isVideo)
        else:
            self.rtsp_path = self.ui.input_camera_rtsp.toPlainText()
            self.demo = Demo(self.rtsp_path, self.output_video, self.output_txt, callback_video=self.callback_video,
                             is_video=self.isVideo)
        self.demo.start()

    def close(self):
        # 关闭事件设为触发，关闭视频播放
        self.stopEvent.set()
        self.ui.open.setEnabled(True)
        self.ui.close.setEnabled(False)
