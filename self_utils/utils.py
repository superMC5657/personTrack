# -*- coding: utf-8 -*-
# !@time: 2020/6/28 下午4:05
# !@author: superMC @email: 18758266469@163.com
# !@fileName: utils.py
import math
import random

import cv2
import numpy as np
import pandas as pd
import torch
from moviepy.video.io.VideoFileClip import VideoFileClip


def get_data(csv_path):
    """
    从csv中拿出face_features 和labels
    """
    name_features_dataframe = pd.read_csv(csv_path, sep=',')
    name_dataframe = name_features_dataframe[['Name']]
    features_name = ['Features%d' % i for i in range(512)]
    features_dataframe = name_features_dataframe[features_name]
    labels = name_dataframe.values
    features = features_dataframe.values
    features = torch.from_numpy(features).type(dtype=torch.float32)
    labels = np.squeeze(labels).tolist()
    print("total_person:", len(labels))
    return labels, features


def self_distance(embeddings1, embeddings2, metric='euclidean'):
    """
    自定义距离
    """
    if metric == 'euclidean_norm':
        # Euclidian distance
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif metric == 'cosine_norm':
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    elif metric == 'euclidean':
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
        return dist
    else:
        raise 'Undefined distance metric %d' % metric

    return dist


def self_compute_distance_matrix(face_features, database_features, metric='euclidean'):
    """
    自定义计算features间距离
    已废弃
    """
    cost_matrix = np.zeros((len(face_features), len(database_features)))
    for i, face_feature in enumerate(face_features):
        cost_matrix[i] = self_distance(face_feature, database_features, metric=metric)
    return cost_matrix


def crop_box(image, box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    return image[y1:y2, x1:x2]


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


def totensor(data):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        data = data.detach()
    return data


def get_color(max_size=100, start=100):
    """因为是黑色面板显示 所以颜色随机区域要亮一点"""
    colors = [tuple(random.randint(start, 255) for _ in range(3)) for _ in range(max_size)]
    return colors


def compute_time(person_caches, record_time):
    """
    计算时间的方法
    如果fps_num > 1 认为人record_time时间内存在
    之后fps_num清零
    """
    for i in range(len(person_caches)):
        if person_caches[i].fps_num > 1:
            person_caches[i].time += record_time
        person_caches[i].fps_num = 0
    return person_caches


def get_video_duration_movie(src_video):
    clip = VideoFileClip(src_video)
    duration = clip.duration
    clip.close()
    return duration


def get_video_duration_cv2(src_video):
    """或许更快"""
    cap = cv2.VideoCapture(src_video)
    if cap.isOpened():
        rate = cap.get(5)
        frame_num = cap.get(7)
        duration = frame_num / rate
        return duration
    return -1


def write_person(person_caches, dst_txt):
    file = open(dst_txt, "w", encoding='utf-8')

    for person in person_caches:
        line = str(person.id) + "\t" + str(person.name) + '\t' + str(person.time) + "\n"
        file.write(line)
    file.close()

