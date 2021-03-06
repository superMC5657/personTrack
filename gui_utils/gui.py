# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets


class Gui:
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1386, 716)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        """
        执行或者结束
        """

        self.open = QtWidgets.QPushButton(self.centralwidget)
        self.open.setGeometry(QtCore.QRect(20, 620, 111, 41))
        self.open.setObjectName("open")

        self.close = QtWidgets.QPushButton(self.centralwidget)
        self.close.setGeometry(QtCore.QRect(200, 620, 111, 41))
        self.close.setObjectName("close")

        """
        确认是否为视频或者是camera
        """

        self.radioButtonVideo = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButtonVideo.setGeometry(QtCore.QRect(20, 450, 96, 41))
        self.radioButtonVideo.setObjectName("radioButtonVideo")

        self.radioButtonCam = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButtonCam.setGeometry(QtCore.QRect(20, 500, 96, 41))
        self.radioButtonCam.setObjectName("radioButtonCam")

        """
        进度条
        """

        self.video_progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.video_progressBar.setGeometry(QtCore.QRect(340, 640, 1041, 23))
        self.video_progressBar.setProperty("value", 0)
        self.video_progressBar.setObjectName("video_progressBar")

        """
        图像显示
        """

        # self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        # self.graphicsView.setGeometry(QtCore.QRect(340, 10, 1041, 621))
        # self.graphicsView.setObjectName("graphicsView")

        self.DispalyLabel = QtWidgets.QLabel(self.centralwidget)
        self.DispalyLabel.setGeometry(QtCore.QRect(340, 10, 1041, 621))
        self.DispalyLabel.setMouseTracking(False)
        self.DispalyLabel.setText("")
        self.DispalyLabel.setObjectName("DispalyLabel")

        """
        show path
        """

        self.input_video_path = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.input_video_path.setGeometry(QtCore.QRect(10, 40, 251, 111))
        self.input_video_path.setObjectName("input_video_path")

        self.input_camera_rtsp = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.input_camera_rtsp.setGeometry(QtCore.QRect(10, 180, 251, 111))
        self.input_camera_rtsp.setPlainText("rtsp://admin:stp123456@10.20.216.46/Streaming/Channels/2")
        self.input_camera_rtsp.setObjectName("input_camera_rtsp")

        self.output_path = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.output_path.setGeometry(QtCore.QRect(10, 320, 251, 111))
        self.output_path.setObjectName("output_path")

        self.find_video = QtWidgets.QPushButton(self.centralwidget)
        self.find_video.setGeometry(QtCore.QRect(260, 40, 61, 111))
        self.find_video.setObjectName("find_video")

        self.find_output = QtWidgets.QPushButton(self.centralwidget)
        self.find_output.setGeometry(QtCore.QRect(260, 320, 61, 111))
        self.find_output.setObjectName("find_output")

        """
        do not care
        """

        self.input_video_path_title = QtWidgets.QLabel(self.centralwidget)
        self.input_video_path_title.setGeometry(QtCore.QRect(10, 20, 141, 18))
        self.input_video_path_title.setObjectName("input_video_path_title")

        self.input_camera_rtsp_title = QtWidgets.QLabel(self.centralwidget)
        self.input_camera_rtsp_title.setGeometry(QtCore.QRect(10, 160, 141, 18))
        self.input_camera_rtsp_title.setObjectName("input_camera_rtsp_title")

        self.output_path_title = QtWidgets.QLabel(self.centralwidget)
        self.output_path_title.setGeometry(QtCore.QRect(10, 300, 141, 18))
        self.output_path_title.setObjectName("output_path_title")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1386, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "VSR"))
        self.open.setText(_translate("MainWindow", "Start"))
        self.close.setText(_translate("MainWindow", "Close"))
        self.radioButtonVideo.setText(_translate("MainWindow", "Offline"))
        self.radioButtonCam.setText(_translate("MainWindow", "Online"))
        self.input_video_path_title.setText(_translate("MainWindow", "Input video path"))
        self.input_camera_rtsp_title.setText(_translate("MainWindow", "Input camera rtsp"))
        self.output_path_title.setText(_translate("MainWindow", "Output path"))
        self.find_video.setText(_translate("MainWindow", "find"))
        self.find_output.setText(_translate("MainWindow", "find"))
