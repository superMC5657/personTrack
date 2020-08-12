# -*- coding: utf-8 -*-
# !@time: 2020/6/10 上午10:59
# !@author: superMC @email: 18758266469@163.com
# !@fileName: image_tool.py

import cv2

from config import opt
from self_utils.utils import colors


def crop_box(image, box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    return image[y1:y2, x1:x2]


def change_coord(landmark_x, landmark_y, x0, y0):
    new_landmark_x = landmark_x - x0
    new_landmark_y = landmark_y - y0
    return new_landmark_x, new_landmark_y


# box1 face box2 person
def person_face_cost(person_box, face_box):
    # print('iou box1:', box1)
    # print('iou box2:', box2)
    ix1 = max(person_box[0], face_box[0])
    ix2 = min(person_box[2], face_box[2])
    iy1 = max(person_box[1], face_box[1])
    iy2 = min(person_box[3], face_box[3])
    iw = max(0, (ix2 - ix1))
    ih = max(0, (iy2 - iy1))
    iarea = iw * ih
    area1 = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
    return 1 - (iarea / area1)


def warp_affine(image, x1, y1, x2, y2, scale=1.0):
    eye_center = ((x1 + x2) / 2, (y1 + y2) / 2)

    dy = y2 - y1
    dx = x2 - x1
    # 计算旋转角度
    angle = cv2.fastAtan2(dy, dx)
    rot = cv2.getRotationMatrix2D(eye_center, angle, scale=scale)

    rot_img = cv2.warpAffine(image, rot, dsize=(image.shape[1], image.shape[0]))

    return rot_img


def plot_boxes(image, persons, fps=25):
    im_height, im_width, _ = image.shape
    scale = 0.6
    persons = sorted(persons, key=lambda x: x.id)
    for i in range(len(persons)):
        pBox = persons[i].pBox
        color = colors[persons[i].id]
        image = cv2.rectangle(image, (pBox[0], pBox[1]), (pBox[2], pBox[3]), color, 2)
        fBox = persons[i].fBox
        if fBox is not None:
            image = cv2.rectangle(image, (fBox[0], fBox[1]), (fBox[2], fBox[3]), color, 2)

        cv2.putText(image, str(persons[i].id) + " " + persons[i].name, (pBox[0] + 5, pBox[3] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color)

        cv2.putText(image, str(persons[i].id) + " " + persons[i].name + " " + str(int(persons[i].fps_num / fps)),
                    (im_width - opt.wight_padding + 5, im_height - 20 * persons[i].id), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color)
    return image
