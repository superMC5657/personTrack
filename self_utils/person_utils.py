# -*- coding: utf-8 -*-
# !@time: 2020/6/28 下午4:03
# !@author: superMC @email: 18758266469@163.com
# !@fileName: person_utils.py
from sklearn.utils.linear_assignment_ import linear_assignment
from torchreid.metrics import compute_distance_matrix
import numpy as np
from torchvision.ops import box_iou

from config import opt

from self_utils.person import Person, Person_Cache
from self_utils.utils import combine_cur_pid, compress_cost_matrix, compute_cost_matrix, \
    combine_cache_pid, get_data, filter_matches_between_people_and_face_frames

database_labels, database_features = get_data(opt.face_data_csv)


def generate_person(person_features, person_boxes, face_features=None, face_boxes=None, face_effective=None,
                    out_face_threshold=opt.out_face_threshold, face_threashold=opt.face_threshold,
                    metric=opt.face_metric):
    """
    根据得到的人脸和人的数据 生成 person 对象
    """

    person_current = [Person(person_features[i], person_boxes[i]) for i in range(len(person_boxes))]
    if face_effective:
        face_names = ['UnKnown' for _ in range(len(face_effective))]
        face_distances = [face_threashold + 0.1 for _ in range(len(face_effective))]

        face_cost_matrix = compute_distance_matrix(face_features, database_features, metric=metric)
        face_matches = linear_assignment(face_cost_matrix)

        for i in range(len(face_matches)):
            a, b = face_matches[i]
            face_distances[a] = face_cost_matrix[a][b].item()
            if face_cost_matrix[a][b] < face_threashold:
                face_names[a] = database_labels[b]

        cost_matrix = compute_cost_matrix(person_boxes, face_boxes)
        filter_line = filter_matches_between_people_and_face_frames(cost_matrix)
        matches = linear_assignment(cost_matrix)

        for i in range(len(matches)):
            a, b = matches[i]
            if a in filter_line:
                continue
            if cost_matrix[a][b] <= out_face_threshold and b in face_effective:
                effective_b = face_effective.index(b)
                person_current[a].fBox = face_boxes[b]
                person_current[a].fid = face_features[effective_b]
                person_current[a].fid_distance = face_distances[effective_b]
                person_current[a].name = face_names[effective_b]
    return person_current


def update_person(person_id, person_current, person_caches, metric=opt.person_metric,
                  person_threshold=opt.person_threshold):
    """
    通过person reid 更新person
    """
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
                person_current[a].update_all(person_caches[b])
                if person_current[a].fid_distance <= person_caches[b].fid_min_distance:
                    person_caches[b].name = person_current[a].name
                    person_caches[b].fid_min_distance = person_current[a].fid_distance
                else:
                    person_current[a].name = person_caches[b].name
                person_current[a].fid_min_distance = person_caches[b].fid_min_distance
                person_caches[b].update_all(person_current[a])

        # 没找到匹配时
        for i in cur_person_dict_notFound:
            person_id += 1
            person_current[i].id = person_id
            person_caches.append(Person_Cache(person_current[i]))

        return person_current, person_caches, person_id

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
    ious = box_iou(person_boxes, person_boxes)
    coincide_index = np.vstack(np.where((ious < 1.0) & (ious > threshold)))
    coincide_index = set(np.unique(coincide_index))
    effective_index = set([i for i in range(person_boxes.shape[0])])
    effective_index = effective_index - coincide_index

    return np.array(list(effective_index))
