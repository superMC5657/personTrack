# -*- coding: utf-8 -*-
# !@time: 2020/6/10 上午10:59
# !@author: superMC @email: 18758266469@163.com
# !@fileName: image_tool.py
import cv2


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
def face_person_cost(box1, box2, x1y1x2y2=True):
    # print('iou box1:', box1)
    # print('iou box2:', box2)

    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

        mx = min(box1[0], box2[0])
        Mx = max(box1[0] + w1, box2[0] + w2)
        my = min(box1[1], box2[1])
        My = max(box1[1] + h1, box2[1] + h2)
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    carea = cw * ch
    return area1 / carea


def warp_affine(image, x1, y1, x2, y2, scale=1.0):
    eye_center = ((x1 + x2) / 2, (y1 + y2) / 2)

    dy = y2 - y1
    dx = x2 - x1
    # 计算旋转角度
    angle = cv2.fastAtan2(dy, dx)
    rot = cv2.getRotationMatrix2D(eye_center, angle, scale=scale)

    rot_img = cv2.warpAffine(image, rot, dsize=(image.shape[1], image.shape[0]))

    return rot_img


def plot_boxes(image, persons):
    for i in range(len(persons)):
        box = persons[i].box
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        scale = (x2 - x1) * 0.005
        cv2.putText(image, str(persons[i].id) + persons[i].name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 255, 0))
    return image
