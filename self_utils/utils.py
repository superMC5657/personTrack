# -*- coding: utf-8 -*-
# !@time: 2020/6/28 下午4:05
# !@author: superMC @email: 18758266469@163.com
# !@fileName: utils.py
import math
import random

import cv2
import pandas as pd
import numpy as np
import torch
from moviepy.video.io.VideoFileClip import VideoFileClip

from config import opt


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


def compute_cost_matrix(person_boxes, face_boxes):
    """
    计算人脸框和人框的代价
    """
    cost_matrix = np.zeros((len(person_boxes), len(face_boxes)))
    for i, person_box in enumerate(person_boxes):
        for j, face_box in enumerate(face_boxes):
            cost_matrix[i][j] = person_face_cost(person_box, face_box)
    return cost_matrix


def compress_cost_matrix(cost_matrix):
    """
    因为person_caches用了多个pid 选取最可信的那个
    """
    person_current_num = cost_matrix.shape[0]
    person_caches_num = int(cost_matrix.shape[1] / opt.cache_len)
    compressed_cost_matrix = torch.zeros((person_current_num, person_caches_num))
    for i in range(person_current_num):
        for j in range(person_caches_num):
            compressed_cost_matrix[i, j] = min(
                cost_matrix[i, j * opt.cache_len:(j + 1) * opt.cache_len])
    return compressed_cost_matrix


def combine_cur_pid(person_current):
    """
    合并person_current 上的pid
    """
    num = len(person_current)
    pids = torch.zeros((num, 512))
    for index, person in enumerate(person_current):
        pids[index] = person.pid
    return pids


def combine_cache_pid(person_caches):
    """
    合并person_cache上的pid_caches
    如果cache_len 与 max_maxLen 一致则认为不使用最后一帧的pid 如果相同则使用
    """
    num = len(person_caches)
    pids = torch.zeros((num * opt.cache_len, 512))
    if opt.cache_len == opt.pid_cache_maxLen:
        for index, person in enumerate(person_caches):
            pids[index * opt.cache_len:
                 (index + 1) * opt.cache_len, :] = person.pid_caches
    else:
        for index, person in enumerate(person_caches):
            pids[index * opt.cache_len:
                 (index + 1) * opt.cache_len - 1, :] = person.pid_caches
            pids[(index + 1) * opt.cache_len - 1, :] = person.pid
    return pids


def person_face_cost(person_box, face_box):
    """
    设计facebox与personbox的代价 用来匹配人脸和人框
    """
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


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


def get_color(max_size=100):
    """因为是黑色面板显示 所以颜色随机区域要亮一点"""
    colors = [tuple(random.randint(155, 255) for _ in range(3)) for _ in range(max_size)]
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
