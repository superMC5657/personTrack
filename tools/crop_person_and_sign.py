# -*- coding: utf-8 -*-
# !@time: 2020/7/20 下午7:25
# !@author: superMC @email: 18758266469@163.com
# !@fileName: crop_person_and_sign.py

import os

import cv2
from torchreid.utils import FeatureExtractor
from pid.yolov4.yolov4 import YoloV4 as Yolo

# from pid.yolov5.yolov5 import YoloV5 as Yolo

if __name__ == '__main__':
    import time

    yolo = Yolo()
    # model = yolov5_model("pid/yolov5/weights/yolov5l_resave.pt")

    image = cv2.imread("data/aoa.jpg")
    # person_images, _ = detect_person(model, image)
    person_images, _ = yolo(image)

    start = time.time()
    for index, person_image in enumerate(person_images):
        image_path = os.path.join("data/crop_persons", chr(index + 65) + ".png")
        cv2.imwrite(image_path, person_image)
        cv2.imshow('demo', person_image)
        cv2.waitKey(0)
