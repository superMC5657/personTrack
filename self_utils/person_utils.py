# -*- coding: utf-8 -*-
# !@time: 2020/6/28 下午4:03
# !@author: superMC @email: 18758266469@163.com
# !@fileName: person_utils.py
import numpy as np
import torch
import zyan
from config import opt
from self_utils.utils import get_data, crop_box, totensor, tonumpy

database_labels, database_features = get_data(opt.face_data_csv)


def compression_person(person_cache):
    """
    Compress person_cache
    去除重复的person合并他们的fps_num
    去除unknown的person
    """
    new_person_cache = []
    name_list = []
    for person in person_cache:
        if person.name is "UnKnown":
            continue
        elif person.name in name_list:
            index = name_list.index(person.name)
            new_person_cache[index].time += person.time
            new_person_cache[index].fps_num += person.fps_num
        else:
            new_person_cache.append(person)
            name_list.append(person.name)

    return new_person_cache


def filter_person(person_boxes, threshold=opt.filter_person_threshold):
    cost = person_face_cost_cpp(person_boxes, person_boxes)
    coincide_index = np.where((cost < 1.0) & (cost > threshold))[1]
    coincide_index = set(np.unique(coincide_index))
    effective_index = set([i for i in range(person_boxes.shape[0])])
    effective_index = effective_index - coincide_index

    return np.array(list(effective_index))


def filter_matches_between_people_and_face_frames(cost_matrix, filter_num=0.0):
    """过滤掉一个人框同时出现多个人脸框"""
    filter_line = []
    for i in range(cost_matrix.shape[0]):
        zero_num = np.where(cost_matrix[i, :] == filter_num)[0].shape[0]
        if zero_num > 1:
            filter_line.append(i)
    return filter_line


def person_face_cost_cpp(person_boxes, face_boxes):
    cost = zyan.person_face_cost(totensor(person_boxes), totensor(face_boxes))
    return tonumpy(cost)


def person_face_cost(person_boxes, face_boxes):
    """
    计算人脸框和人框的代价
    """
    cost_matrix = np.zeros((len(person_boxes), len(face_boxes)))
    for i, person_box in enumerate(person_boxes):
        for j, face_box in enumerate(face_boxes):
            cost_matrix[i][j] = self_cost_func(person_box, face_box)
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


def self_cost_func(person_box, face_box):
    """
    设计facebox与personbox的代价 用来匹配人脸和人框
    """
    # print('iou box1:', box1)
    # print('iou box2:', box2)
    ix1 = max(person_box[0], face_box[0])
    ix2 = min(person_box[2], face_box[2])
    iy1 = max(person_box[1], face_box[1])
    iy2 = min(person_box[3], face_box[3])
    iw = max(0, (ix2 - ix1 + 1))
    ih = max(0, (iy2 - iy1 + 1))
    iarea = iw * ih
    farea = (face_box[2] - face_box[0] + 1) * (face_box[3] - face_box[1] + 1)
    return iarea / farea


def crop_persons(image, person_boxes):
    person_boxes = person_boxes.astype(np.int32)
    person_effective = filter_person(person_boxes)
    if len(person_effective) == 0:
        return [], []
    person_images = []
    person_boxes = person_boxes[person_effective]
    for xyxy in person_boxes:
        person_images.append(crop_box(image, xyxy))
    return person_images, person_boxes
