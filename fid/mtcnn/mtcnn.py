# -*- coding: utf-8 -*-
# !@time: 2020/7/13 下午2:33
# !@author: superMC @email: 18758266469@163.com
# !@fileName: mtcnn.py
import cv2

from fid.mtcnn.detect import create_mtcnn_net, MtcnnDetector
import numpy as np
from self_utils.image_tool import crop_box, change_coord, warp_affine


class MTCNN(object):
    def __init__(self, p_model_path="fid/mtcnn/mtcnn_checkpoints/pnet_epoch.pt",
                 r_model_path="fid/mtcnn/mtcnn_checkpoints/rnet_epoch.pt",
                 o_model_path="fid/mtcnn/mtcnn_checkpoints/onet_epoch.pt", use_cuda=1, min_face_size=24):
        pnet, rnet, onet = create_mtcnn_net(p_model_path, r_model_path, o_model_path, use_cuda)
        self.detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=min_face_size)

    def __call__(self, image):
        # start = time.time()
        # for i in range(100):
        #     bboxs, landmarks = detector.detect_face(image)
        # print(time.time() - start)
        bboxs, landmarks = self.detector.detect_face(image)
        width = image.shape[1]
        height = image.shape[0]
        face_boxes = []
        faces = []
        for box, landmark in zip(bboxs, landmarks):
            box[0] = np.maximum(box[0], 0)
            box[1] = np.maximum(box[1], 0)
            box[2] = np.minimum(box[2], width)  # w
            box[3] = np.minimum(box[3], height)

            face = crop_box(image, box)
            eye_left_x, eye_left_y = change_coord(landmark[0], landmark[1], box[0], box[1])
            eye_right_x, eye_right_y = change_coord(landmark[2], landmark[3], box[0], box[1])
            try:
                face = warp_affine(image=face, x1=eye_left_x, y1=eye_left_y, x2=eye_right_x,
                                   y2=eye_right_y)
            except:
                print(box, landmark)
                cv2.imwrite('data/error.png', image)
            faces.append(face)
            face_boxes.append([int(_) for _ in box[:4]])
        return faces, face_boxes


if __name__ == '__main__':
    mtcnn = MTCNN()
    image = cv2.imread('data/office1.jpg')
    faces, boxes = mtcnn(image)
    for box in boxes:
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    cv2.imshow('demo', image)
    cv2.waitKey(0)
