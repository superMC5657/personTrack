# -*- coding: utf-8 -*-
# !@time: 2020/7/16 下午1:53
# !@author: superMC @email: 18758266469@163.com
# !@fileName: config.py
from pprint import pprint


class Config:
    person_threshold = 200  # 越小越严格
    face_threshold = 1.1  # 越小越严格
    filter_face_threshold = 0.42  # 人脸偏转幅度(双眼占比人脸宽度 >阈值生效) 越大越严格
    filter_person_threshold = 0.5  # 单个person与其他person重合百分比 < 阈值生效 越小越严格
    out_face_threshold = 0.0  # 人脸和人/人脸 百分比 越小越严格

    face_metric = 'euclidean'
    person_metric = 'euclidean'
    face_data_csv = 'data/4329.csv'

    default_output_name = "output"

    wight_padding = 220

    vis_video = False
    show_fps = 0  # 10
    video_speed = 4  # 最好小于5

    compress_time = 1000
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
