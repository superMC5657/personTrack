# -*- coding: utf-8 -*-
# !@time: 2020/7/20 下午7:21
# !@author: superMC @email: 18758266469@163.com
# !@fileName: crop_face_and_sign.py

import os

import cv2
import torch
from torch.backends import cudnn

from fid.insightFace.faceNet import FaceNet
from fid.mtcnn.mtcnn import MTCNN
from fid.retinaFace.detector import Detector as RetinaFace

cudnn.benchmark = True
torch.set_grad_enabled(False)
if __name__ == '__main__':
    retinaFace_detector = RetinaFace()
    # mtcnn_detector = MTCNN()
    # faceModel = FaceNet()
    img = cv2.imread("data/IZTY.png")
    # faces, _ = retinaFace_detector(img)
    faces, _ = retinaFace_detector.forward_for_makecsv(img)
    for id, face in enumerate(faces):
        image_path = os.path.join("data/face_with_name", chr(id + 65) + ".png")
        cv2.imwrite(image_path, face)
        cv2.imshow('demo', face)
        cv2.waitKey(0)
