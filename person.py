# -*- coding: utf-8 -*-
# !@time: 2020/6/10 下午10:28
# !@author: superMC @email: 18758266469@163.com
# !@fileName: person.py
from self_utils.compare import compare_feature, get_data

labels, old_features = get_data("data/one_man_img.csv")


class Person(object):
    def __init__(self):
        self.id = 0
        self.name = 'UnKnown'
        self.fid = None  # face features
        self.pid = None  # reid features
        self.box = None  # bbox
        self.time = None  # 统计帧数
        self.fid_min_distance = None  # (min_distance, name)

    def findOut_name(self):
        if self.fid is not None:
            name = compare_feature(self.fid, old_features, labels)[0]
            self.name = self.name if name == 'UnKnown' else name

    def update_name(self):
        pass
