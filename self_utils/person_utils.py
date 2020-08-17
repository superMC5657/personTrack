# -*- coding: utf-8 -*-
# !@time: 2020/6/28 下午4:03
# !@author: superMC @email: 18758266469@163.com
# !@fileName: person_utils.py
from sklearn.utils.linear_assignment_ import linear_assignment
from torchreid.metrics import compute_distance_matrix

from config import opt
from self_utils.person import Person, database_features, database_labels, Person_Cache
from self_utils.utils import combine_cur_pid, compress_cost_matrix, compute_cost_matrix, \
    combine_cache_pid


def generate_person(person_features, person_boxes, face_features=None, face_boxes=None, threshold=opt.threshold,
                    face_threashold=opt.face_threshold, metric=opt.face_metric):
    # face_list = [_ for _ in range(len(face_boxes))]
    # person_list = [_ for _ in range(len(person_boxes))]
    person_current = [Person(person_features[i], person_boxes[i]) for i in range(len(person_boxes))]
    face_names = ['UnKnown' for _ in range(len(face_boxes))]
    face_distances = [face_threashold + 0.1 for _ in range(len(face_boxes))]
    if face_names:
        # print("faces:{}, persons:{}".format(len(face_boxes), len(person_boxes)))
        face_cost_matrix = compute_distance_matrix(face_features, database_features, metric=metric)
        face_matches = linear_assignment(face_cost_matrix)
        for i in range(len(face_matches)):
            a, b = face_matches[i]
            face_distances[a] = face_cost_matrix[a][b].item()
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
                person_current[a].fid_distance = face_distances[b]
                person_current[a].name = face_names[b]
    return person_current


# 通过person reid 更新person
def update_person(person_id, person_current, person_caches, metric=opt.person_metric,
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
            new_person_cache[index].time += person.time
            new_person_cache[index].fps_num += person.fps_num
        else:
            new_person_cache.append(person)
            name_list.append(person.name)

    return new_person_cache
