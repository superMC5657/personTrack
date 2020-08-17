# -*- coding: utf-8 -*-
# !@time: 2020/7/16 下午1:53
# !@author: superMC @email: 18758266469@163.com
# !@fileName: config.py
from pprint import pprint


class Config:
    person_threshold = 200
    face_threshold = 1.1
    face_metric = 'euclidean'
    person_metric = 'euclidean'
    threshold = 0.5

    wight_padding = 250
    is_video = True
    vis = True
    video_speed = 1

    compress_time = 60
    record_time = 1

    pid_cache_maxLen = 10
    cache_len = 10  # 上一帧
    pid_stride = 3  # 1 frame per save

    def parse(self, kwargs):
        state_dict = self.state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self.state_dict())
        print('==========end============')

    def state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
