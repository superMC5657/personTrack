# -*- coding: utf-8 -*-
# !@time: 2020/6/28 下午4:03
# !@author: superMC @email: 18758266469@163.com
# !@fileName: person_utils.py
import numpy as np
import torch
from sklearn.utils.linear_assignment_ import linear_assignment
from torchreid.metrics import compute_distance_matrix
from config import opt
from self_utils.person import Person, database_features, database_labels, Person_Cache
from self_utils.image_tool import person_face_cost
from self_utils.utils import self_compute_distance_matrix


def compute_cost_matrix(person_boxes, face_boxes):
    cost_matrix = np.zeros((len(person_boxes), len(face_boxes)))
    for i, person_box in enumerate(person_boxes):
        for j, face_box in enumerate(face_boxes):
            cost_matrix[i][j] = person_face_cost(person_box, face_box)
    return cost_matrix


def combine_cur_pid(person_current):
    num = len(person_current)
    pids = torch.zeros((num, 512))
    for index, person in enumerate(person_current):
        pids[index] = person.pid
    return pids


def generate_person(person_features, person_boxes, face_features=None, face_boxes=None, threshold=opt.threshold,
                    face_threashold=opt.face_threshold):
    # face_list = [_ for _ in range(len(face_boxes))]
    # person_list = [_ for _ in range(len(person_boxes))]
    person_current = [Person(person_features[i], person_boxes[i]) for i in range(len(person_boxes))]
    face_names = ['UnKnown' for _ in range(len(face_boxes))]
    if face_names:
        # print("faces:{}, persons:{}".format(len(face_boxes), len(person_boxes)))
        face_cost_matrix = self_compute_distance_matrix(face_features, database_features, distance_metric=2)
        face_matches = linear_assignment(face_cost_matrix)
        for i in range(len(face_matches)):
            a, b = face_matches[i]
            if face_cost_matrix[a][b] < face_threashold:
                face_names[a] = database_labels[b]
        cost_matrix = compute_cost_matrix(person_boxes, face_boxes)
        matches = linear_assignment(cost_matrix)
        for i in range(len(matches)):
            a, b = matches[i]
            if cost_matrix[a][b] < threshold:
                # person_list.remove(a)
                # face_list.remove(b)
                person_current[a].fBox = face_boxes[b]
                person_current[a].fid = face_features[b]
                person_current[a].name = face_names[b]
    return person_current


def combine_cache_pid(person_caches):
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


def compress_cost_matrix(cost_matrix):
    person_current_num = cost_matrix.shape[0]
    person_caches_num = int(cost_matrix.shape[1] / opt.cache_len)
    compressed_cost_matrix = torch.zeros((person_current_num, person_caches_num))
    for i in range(person_current_num):
        for j in range(person_caches_num):
            compressed_cost_matrix[i, j] = min(
                cost_matrix[i, j * opt.cache_len:(j + 1) * opt.cache_len])
    return compressed_cost_matrix


# 通过person reid 更新person
def update_person(person_id, person_current, person_caches, metric='euclidean',
                  person_threshold=opt.person_threshold):
    # cost_matrix = pw.pairwise_distances(combine_pid(person_cache), combine_pid(person_current))
    # 当cache不存在时
    if not person_caches:
        for person in person_current:
            person_id += 1
            person.id = person_id
            person_caches.append(Person_Cache(person))
        return person_current, person_caches, person_id

    else:
        cost_matrix = compute_distance_matrix(combine_cur_pid(person_current), combine_cache_pid(person_caches),
                                              metric=metric)
        cost_matrix = compress_cost_matrix(cost_matrix)
        matches = linear_assignment(cost_matrix)
        cur_person_dict_notFound = [i for i in range(len(person_current))]
        for i in range(len(matches)):
            a, b = matches[i]
            if cost_matrix[a][b] < person_threshold:
                cur_person_dict_notFound.remove(a)
                person_current[a].id = person_caches[b].id
                if person_current[a].name is "UnKnown" and person_caches[b].name is not "UnKnown":
                    person_current[a].name = person_caches[b].name
                person_caches[b].fps_num += 1
                person_current[a].fps_num = person_caches[b].fps_num
                person_caches[b].update_all(person_current[a])

        # 没找到匹配时
        for i in cur_person_dict_notFound:
            person_id += 1
            person_current[i].id = person_id
            person_caches.append(Person_Cache(person_current[i]))

        return person_current, person_caches, person_id


def update_person_caches():
    pass


def update_person_current():
    pass


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
            new_person_cache[index].fps_num += person.fps_num
        else:
            new_person_cache.append(person)
            name_list.append(person.name)

    return new_person_cache
