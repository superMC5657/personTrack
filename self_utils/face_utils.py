# -*- coding: utf-8 -*-
# !@time: 19-4-28 下午10:06
# !@author: superMC @email: 18758266469@163.com
# !@fileName: csv_features.py

import cv2
import numpy as np

from config import opt
from fid.demo import FaceNet
from fid.retinaFace.detector import Detector as RetinaFace
from self_utils.utils import get_data, distance


# 比较csv中的features和传入新人脸的features
def get_names(new_features, old_features, labels, threshold=opt.face_threshold):
    # distances = pw.pairwise_distances(new_features, old_features, metric)
    distances = distance(new_features, old_features, distance_metric=2)
    names = []
    for i in range(len(new_features)):
        min_distance_index = np.argmin(distances)
        if distances[min_distance_index] < threshold:
            names.append(labels[min_distance_index])
        else:
            names.append('UnKnown')
    return names


# 根据features 找到超过阀值的索引所对应的NAME
def main(detector, faceNet, labels, old_features, img, threshold=opt.face_threshold):
    faces, _ = detector(img)
    if len(faces) == 0:
        return None
    names = []
    for face in faces:
        new_features = faceNet(face)
        name = get_names(new_features, old_features, labels, threshold=threshold)[0]
        names.append(name)
    return names


if __name__ == '__main__':
    # detector = MTCNN()
    csv_path = 'data/one_man_img.csv'
    img_path = '/home/supermc/Pictures/IZTY.png'
    detector = RetinaFace()
    mobileFace = FaceNet("fid/insightFace/facenet_checkpoints/model_ir_se50.pth")
    labels, old_features = get_data(csv_path)
    img = cv2.imread(img_path)
    print(main(detector, mobileFace, labels, old_features, img))
