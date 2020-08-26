# -*- coding: utf-8 -*-
# !@time: 2020/8/23 下午7:33
# !@author: superMC @email: 18758266469@163.com
# !@fileName: face_utils.py
import cv2
import numpy as np

from config import opt
from self_utils.utils import crop_box


def change_coord(landmark_x, landmark_y, x0, y0):
    new_landmark_x = landmark_x - x0
    new_landmark_y = landmark_y - y0
    return new_landmark_x, new_landmark_y


def warp_affine(image, x1, y1, x2, y2, scale=1.0):
    eye_center = ((x1 + x2) / 2, (y1 + y2) / 2)

    dy = y2 - y1
    dx = x2 - x1
    # 计算旋转角度
    angle = cv2.fastAtan2(dy, dx)
    rot = cv2.getRotationMatrix2D(eye_center, angle, scale=scale)

    rot_img = cv2.warpAffine(image, rot, dsize=(image.shape[1], image.shape[0]))

    return rot_img


def crop_faces(image, boxes, landms):
    width = image.shape[1]
    height = image.shape[0]
    boxes[:, 0] = np.maximum(boxes[:, 0], 0)
    boxes[:, 1] = np.maximum(boxes[:, 1], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], width)  # w
    boxes[:, 3] = np.minimum(boxes[:, 3], height)
    faces = list()
    face_effective = list()
    for index, (box, landm) in enumerate(zip(boxes, landms)):
        eye_left_x, eye_left_y = change_coord(landm[0], landm[1], box[0], box[1])
        eye_right_x, eye_right_y = change_coord(landm[2], landm[3], box[0], box[1])
        box_width = box[2] - box[0]
        eye_width = eye_right_x - eye_left_x
        if eye_width / box_width > opt.filter_face_threshold:
            face = crop_box(image, box)
            face = warp_affine(image=face, x1=eye_left_x, y1=eye_left_y, x2=eye_right_x, y2=eye_right_y)
            faces.append(face)
            face_effective.append(index)
    boxes = boxes.astype(np.int32)
    return faces, boxes, face_effective
