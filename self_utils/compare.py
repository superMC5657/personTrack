# -*- coding: utf-8 -*-
# !@time: 19-4-28 下午10:06
# !@author: superMC @email: 18758266469@163.com
# !@fileName: csv_features.py
import math
import os
import warnings

import cv2
import torch
from sklearn.utils.linear_assignment_ import linear_assignment
from torchreid.metrics import compute_distance_matrix

from fid.mtcnn.mtcnn import MTCNN
from fid.retinaFace.detector import Detector as RetinaFace
from self_utils.utils import get_data

warnings.filterwarnings('ignore')
os.environ['GLOG_minloglevel'] = '3'
from fid.demo import FaceNet
import numpy as np
import copy


# 从csv文件中获取姓名与其对应的特征


def distance(embeddings1, embeddings2, distance_metric=2):
    if distance_metric == 0:
        # Euclidian distance
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    elif distance_metric == 2:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
        return dist
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist


# 比较csv中的features和传入新人脸的features
def compare_feature(new_features, old_features, labels, threshold=1.2):
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
def get_names(detector, faceNet, labels, old_features, img, threshold=1.2):
    faces, _ = detector(img)
    if len(faces) == 0:
        return None
    names = []
    for face in faces:
        new_features = faceNet(face)
        name = compare_feature(new_features, old_features, labels, threshold=threshold)[0]
        names.append(name)
    return names


def combine_pid(persons):
    num = len(persons)
    pids = torch.zeros((num, 512))
    for index, person in enumerate(persons):
        pids[index] = person.pid
    return pids


# 通过person reid 更新person
def update_person(index, person_cache: list, cur_person_dict, metric='euclidean', person_threshold=120):
    # cost_matrix = pw.pairwise_distances(combine_pid(person_cache), combine_pid(cur_person_dict))
    # cost_matrix = distance(combine_pid(person_cache), combine_pid(cur_person_dict))

    # 当cache不存在时
    if not person_cache:
        for person in cur_person_dict:
            index += 1
            person.id = index
            person_cache.append(person)
        return person_cache, person_cache, index
    else:
        cost_matrix = compute_distance_matrix(combine_pid(cur_person_dict), combine_pid(person_cache), metric=metric)
        matches = linear_assignment(cost_matrix)
        new_person_cache = copy.deepcopy(person_cache)
        cur_person_dict_notFound = [i for i in range(len(cur_person_dict))]
        for i in range(len(matches)):
            a, b = matches[i]
            if cost_matrix[a][b] < person_threshold:
                cur_person_dict_notFound.remove(a)
                cur_person_dict[a].id = person_cache[b].id
                if cur_person_dict[a].name is "UnKnown" and person_cache[b].name is not "UnKnown":
                    cur_person_dict[a].name = person_cache[b].name
                new_person_cache[b] = cur_person_dict[a]

        # 没找到匹配时
        for i in cur_person_dict_notFound:
            index += 1
            cur_person_dict[i].id = index
            new_person_cache.append(cur_person_dict[i])

        return new_person_cache, cur_person_dict, index


def main(csv_path, img_path):
    # detector = MTCNN()
    detector = RetinaFace()
    mobileFace = FaceNet("fid/insightFace/facenet_checkpoints/model_ir_se50.pth")
    labels, old_features = get_data(csv_path)
    img = cv2.imread(img_path)
    print(get_names(detector, mobileFace, labels, old_features, img))


if __name__ == '__main__':
    main(csv_path='data/one_man_img.csv', img_path='/home/supermc/Pictures/IZTY.png')
