# -*- coding: utf-8 -*-
# !@time: 2020/10/15 下午4:06
# !@author: superMC @email: 18758266469@163.com
# !@fileName: models.py
from torchreid.utils import FeatureExtractor

from fid.insightFace.faceNet import FaceNet
from fid.retinaFace.detector import Detector as RetinaFace
from pid.yolov5.yolov5 import YoloV5


def init_models(video_size=(704,576)):
    yolo = YoloV5()
    reid = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='pid/deep_person_reid/checkpoints/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
        verbose=False)
    face_detector = RetinaFace(image_size=(video_size[1], video_size[0]))
    # detector = MTCNN()
    faceNet = FaceNet()
    return yolo, reid, face_detector, faceNet
