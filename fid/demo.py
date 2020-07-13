# -*- coding: utf-8 -*-
# !@time: 2020/6/10 上午9:27
# !@author: superMC @email: 18758266469@163.com
# !@fileName: demo.py
import os

import cv2
from fid.insightFace.faceNet import FaceNet
from fid.mtcnn.mtcnn import MTCNN
from fid.retinaFace.detector import Detector as RetinaFace

if __name__ == '__main__':
    retinaFace_detector = RetinaFace()
    # mtcnn_detector = MTCNN()
    faceModel = FaceNet()
    img = cv2.imread("data/aoa.jpg")
    # faces, _ = retinaFace_detector(img)
    faces, _ = retinaFace_detector(img)
    for id, face in enumerate(faces):
        image_path = os.path.join("data/face_with_name", chr(id + 65) + ".png")
        cv2.imwrite(image_path, face)
        cv2.imshow('demo', face)
        cv2.waitKey(0)
    features = faceModel(faces)
    print(features)
