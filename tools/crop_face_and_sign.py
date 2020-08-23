# -*- coding: utf-8 -*-
# !@time: 2020/7/20 下午7:21
# !@author: superMC @email: 18758266469@163.com
# !@fileName: crop_face_and_sign.py

import os

import cv2
import torch

from fid.insightFace.faceNet import FaceNet
from fid.mtcnn.mtcnn import MTCNN
from fid.retinaFace.detector import Detector as RetinaFace

torch.set_grad_enabled(False)
if __name__ == '__main__':
    face_detector = RetinaFace(pre_size=False)
    # mtcnn_detector = MTCNN()
    # faceModel = FaceNet()
    img = cv2.imread("data/office1.jpg")
    # faces, _ = retinaFace_detector(img)
    faces, *_ = face_detector(img)
    for id, face in enumerate(faces):
        image_path = os.path.join("data/crop_faces", chr(id + 65) + ".png")
        cv2.imwrite(image_path, face)
        cv2.imshow('demo', face)
        cv2.waitKey(0)
