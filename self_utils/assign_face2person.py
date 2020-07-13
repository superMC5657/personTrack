# -*- coding: utf-8 -*-
# !@time: 2020/6/28 下午4:03
# !@author: superMC @email: 18758266469@163.com
# !@fileName: assign_face2person.py
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

from person import Person
from self_utils.image_tool import person_face_cost


def compute_cost_matrix(person_boxes, face_boxes):
    cost_matrix = np.zeros((len(person_boxes), len(face_boxes)))
    for i, person_box in enumerate(person_boxes):
        for j, face_box in enumerate(face_boxes):
            cost_matrix[i][j] = person_face_cost(person_box, face_box)
    return cost_matrix


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
