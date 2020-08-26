# -*- coding: utf-8 -*-
# !@time: 2020/6/26 下午5:56
# !@author: superMC @email: 18758266469@163.com
# !@fileName: __init__.py
import cv2
import numpy as np

from pid.yolov4.tool.darknet2pytorch import Darknet
from pid.yolov4.tool.torch_utils import do_detect
from self_utils.person_utils import crop_persons
from self_utils.utils import totensor


class YoloV4:
    def __init__(self, cfg="pid/yolov4/cfg/yolov4.cfg", weight="pid/yolov4/yolov4_checkpoints/yolov4.weights",
                 use_cuda=1):
        model = Darknet(cfg)

        # model.print_network()
        model.load_weights(weight)
        # 1print('Loading weights from %s... Done!' % (weight))
        if use_cuda:
            model.cuda()
        model.eval()
        self.model = model
        del model

    def __call__(self, image):
        rgb_image = cv2.resize(image, (self.model.width, self.model.height))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        boxes = do_detect(self.model, rgb_image, 0.5, 80, 0.4)
        if boxes is None:
            return [], []
        boxes = np.vstack(boxes)
        person_indices = np.where(boxes[:, 6] == 0)[0]
        boxes = boxes[person_indices]
        width = image.shape[1]
        height = image.shape[0]
        new_boxes = np.zeros((boxes.shape[0], 4))
        new_boxes[:, 0] = np.maximum((boxes[:, 0] - boxes[:, 2] / 2.0) * width, 0)
        new_boxes[:, 1] = np.maximum((boxes[:, 1] - boxes[:, 3] / 2.0) * height, 0)
        new_boxes[:, 2] = np.minimum((boxes[:, 0] + boxes[:, 2] / 2.0) * width, width)  # w
        new_boxes[:, 3] = np.minimum((boxes[:, 1] + boxes[:, 3] / 2.0) * height, height)
        return crop_persons(image, new_boxes)


if __name__ == '__main__':
    model = YoloV4()
    image = cv2.imread('data/aoa.jpg')
    person_images, boxes = model(image)
    for index, person_image in enumerate(person_images):
        cv2.imshow('demo', person_image)
        cv2.waitKey(0)
