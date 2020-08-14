# -*- coding: utf-8 -*-
# !@time: 2020/6/10 下午10:28
# !@author: superMC @email: 18758266469@163.com
# !@fileName: person.py
import torch

from config import opt
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

        self.face_distance = None

    def update_name(self):
        pass


class Person_Cache:
    def __init__(self, person):
        self.id = None
        self.name = None
        self.fps_num = None
        self.pid = None
        self.replace_index = 0
        self.pid_caches = torch.zeros((opt.pid_cache_maxLen, 512))
        self.fid_min_distance = None  # (name,min distance)
        self.update_base(person)
        self.update_pid_caches(person)

    def update_pid_caches(self, person):
        self.pid_caches[self.replace_index * opt.pid_cache_maxLen:
                        (self.replace_index + 1) * opt.pid_cache_maxLen, :] = person.pid
        self.replace_index += 1
        self.replace_index %= opt.pid_stride

    def update_base(self, person):
        self.id = person.id
        self.name = person.name
        self.fps_num = person.fps_num
        self.pid = person.pid

    def update_all(self, person):
        self.update_base(person)
        if self.fps_num % opt.pid_stride == 0:
            self.update_pid_caches(person)
