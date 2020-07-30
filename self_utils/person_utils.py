# -*- coding: utf-8 -*-
# !@time: 2020/6/28 下午4:03
# !@author: superMC @email: 18758266469@163.com
# !@fileName: person_utils.py
import numpy as np
import torch
from sklearn.utils.linear_assignment_ import linear_assignment
from torchreid.metrics import compute_distance_matrix
from config import opt
from self_utils.person import Person
from self_utils.image_tool import person_face_cost


def compute_cost_matrix(person_boxes, face_boxes):
    cost_matrix = np.zeros((len(person_boxes), len(face_boxes)))
    for i, person_box in enumerate(person_boxes):
        for j, face_box in enumerate(face_boxes):
            cost_matrix[i][j] = person_face_cost(person_box, face_box)
    return cost_matrix


def combine_pid(persons):
    num = len(persons)
    pids = torch.zeros((num, 512))
    for index, person in enumerate(persons):
        pids[index] = person.pid
    return pids


def generate_person(person_features, person_boxes, face_features=None, face_boxes=None, threashold=0.5):
    # face_list = [_ for _ in range(len(face_boxes))]
    # person_list = [_ for _ in range(len(person_boxes))]
    cur_person_dict = [Person(person_features[i], person_boxes[i]) for i in range(len(person_boxes))]
    if face_boxes is not None:
        # print("faces:{}, persons:{}".format(len(face_boxes), len(person_boxes)))
        cost_matrix = compute_cost_matrix(person_boxes, face_boxes)
        matches = linear_assignment(cost_matrix)
        for i in range(len(matches)):
            a, b = matches[i]
            if cost_matrix[a][b] < threashold:
                # person_list.remove(a)
                # face_list.remove(b)
                cur_person_dict[a].fBox = face_boxes[b]
                cur_person_dict[a].fid = face_features[b]
                cur_person_dict[a].findOut_name()

    return cur_person_dict


# 通过person reid 更新person
def update_person(index, person_cache: list, cur_person_dict, metric='euclidean',
                  person_threshold=opt.person_threshold):
    # cost_matrix = pw.pairwise_distances(combine_pid(person_cache), combine_pid(cur_person_dict))
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
        cur_person_dict_notFound = [i for i in range(len(cur_person_dict))]
        for i in range(len(matches)):
            a, b = matches[i]
            if cost_matrix[a][b] < person_threshold:
                cur_person_dict_notFound.remove(a)
                cur_person_dict[a].id = person_cache[b].id
                if cur_person_dict[a].name is "UnKnown" and person_cache[b].name is not "UnKnown":
                    cur_person_dict[a].name = person_cache[b].name
                cur_person_dict[a].fps_num = person_cache[b].fps_num
                person_cache[b] = cur_person_dict[a]
                person_cache[b].fps_num += 1

        # 没找到匹配时
        for i in cur_person_dict_notFound:
            index += 1
            cur_person_dict[i].id = index
            person_cache.append(cur_person_dict[i])

        return person_cache, cur_person_dict, index


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
