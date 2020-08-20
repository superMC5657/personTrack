# -*- coding: utf-8 -*-
# !@time: 2020/6/9 上午5:42
# !@author: superMC @email: 18758266469@163.com
# !@fileName: demo.py
import os

import cv2
from torchreid.utils import FeatureExtractor
from pid.yolov4.yolov4 import YoloV4 as Yolo

# from pid.yolov5.yolov5 import YoloV5 as Yolo

if __name__ == '__main__':
    import time

    yolo = Yolo()
    # model = yolov5_model("pid/yolov5/weights/yolov5l_resave.pt")
    reid = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='pid/deep_person_reid/checkpoints/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
        verbose=False)

    image = cv2.imread("data/office1.jpg")
    # person_images, _ = detect_person(model, image)
    person_images, _ = yolo(image)

    start = time.time()
    for index, person_image in enumerate(person_images):
        cv2.imshow('demo', person_image)
        cv2.waitKey(0)
    features = reid(person_images)
    print(features.shape)
    print(time.time() - start)
