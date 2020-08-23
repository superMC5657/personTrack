# -*- coding: utf-8 -*-
# !@time: 2020/6/10 上午9:27
# !@author: superMC @email: 18758266469@163.com
# !@fileName: fid_demo.py

import cv2
import torch

from fid.mtcnn.mtcnn import MTCNN
from fid.mtcnn_caffe.mtcnn import MTCNN as MTCNN_CAFFE
from fid.retinaFace.detector import Detector as RetinaFace

torch.set_grad_enabled(False)
if __name__ == '__main__':
    face_detector = RetinaFace(pre_size=False)
    face_detector = MTCNN_CAFFE()
    face_detector = MTCNN(min_face_size=10)
    image = cv2.imread('data/office1.jpg')
    faces, boxes, _ = face_detector(image)
    for box in boxes:
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    cv2.imshow('demo', image)
    cv2.waitKey(0)
