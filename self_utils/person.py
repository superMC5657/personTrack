# -*- coding: utf-8 -*-
# !@time: 2020/6/10 下午10:28
# !@author: superMC @email: 18758266469@163.com
# !@fileName: person.py
from self_utils.face_utils import get_data

database_labels, database_features = get_data("data/one_man_img.csv")


class Person(object):
    def __init__(self, pid, pBox):
        self.id = 0
        self.name = 'UnKnown'
        self.fid = None  # face features
        self.pid = pid  # reid features
        self.pBox = pBox  # person bbox
        self.fBox = None  # face bbox
        self.fps_num = 1
        self.fid_min_distance = None  # (min_distance, name)

    def update_name(self):
        pass
