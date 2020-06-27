# -*- coding: utf-8 -*-
# !@time: 2020/6/9 上午5:42
# !@author: superMC @email: 18758266469@163.com
# !@fileName: face_model.py
import os

import cv2
from torchreid.utils import FeatureExtractor
# from pid.yolov4.yolov4 import yolov4_model as yolo_model
# from pid.yolov4.yolov4 import detect_person
from pid.yolov5.yolov5 import yolov5_model as yolo_model
from pid.yolov5.yolov5 import detect_person

if __name__ == '__main__':
    import time

    model = yolo_model()
    # model = yolov5_model("pid/yolov5/weights/yolov5l_resave.pt")
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='deep_person_reid/checkpoints/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
    )

    image = cv2.imread("data/aoa.jpg")
    # person_images, _ = detect_person(model, image)
    person_images, _ = detect_person(model, image)

    start = time.time()
    for index, person_image in enumerate(person_images):
        features = extractor(person_image)
        image_path = os.path.join("data/crop_persons", chr(index + 65) + ".png")
        cv2.imwrite(image_path, person_image)
        cv2.imshow('demo', person_image)
        cv2.waitKey(0)
        print(features.shape)
    print(time.time() - start)
