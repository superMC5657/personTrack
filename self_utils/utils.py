# -*- coding: utf-8 -*-
# !@time: 2020/6/28 下午4:05
# !@author: superMC @email: 18758266469@163.com
# !@fileName: utils.py
import math
import random
import pandas as pd
import numpy as np
import torch


def get_data(csv_path):
    name_features_dataframe = pd.read_csv(csv_path, sep=',')
    name_dataframe = name_features_dataframe[['Name']]
    features_name = ['Features%d' % i for i in range(512)]
    features_dataframe = name_features_dataframe[features_name]
    labels = name_dataframe.values
    features = features_dataframe.values
    labels = np.squeeze(labels).tolist()
    return labels, features


def one_distance(embeddings1, embeddings2, distance_metric=2):
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


def self_compute_distance_matrix(face_features, database_features, distance_metric=2):
    cost_matrix = np.zeros((len(face_features), len(database_features)))
    for i, face_feature in enumerate(face_features):
        cost_matrix[i] = one_distance(face_feature, database_features, distance_metric=distance_metric)
    return cost_matrix


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


def get_color(max_size=100):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(max_size)]
    return colors
