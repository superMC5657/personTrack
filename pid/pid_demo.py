# -*- coding: utf-8 -*-
# !@time: 2020/6/9 上午5:42
# !@author: superMC @email: 18758266469@163.com
# !@fileName: pid_demo.py

import cv2
from torchreid.utils import FeatureExtractor

from pid.yolov4.yolov4 import YoloV4 as Yolo

#from pid.yolov5.yolov5 import YoloV5 as Yolo

if __name__ == '__main__':
    import time

    yolo = Yolo()
    reid = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='pid/deep_person_reid/checkpoints/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
        verbose=False)

    image = cv2.imread("data/office1.jpg")
    # person_images, _ = detect_person(model, image)
    person_images, person_boxes = yolo(image)
    for box in person_boxes:
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    cv2.imshow('demo', image)
    cv2.waitKey(0)
