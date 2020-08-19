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
    face_data_csv = 'data/1692.csv'

    wight_padding = 200
    is_video = True
    vis_video = False
    show_fps = False
    video_speed = 1

    compress_time = 60
    record_time = 1
    callback_time = 1

    pid_cache_maxLen = 10  # pid_caches最大长度
    cache_len = 10  # 如果cache_len == pid_cache_maxLen 则不使用最后一帧做reid 若果cache_len == pid_cache_maxLen +1 就是使用最后一帧做reid
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
