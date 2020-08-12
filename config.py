# -*- coding: utf-8 -*-
# !@time: 2020/7/16 下午1:53
# !@author: superMC @email: 18758266469@163.com
# !@fileName: config.py
from pprint import pprint


class Config:
    person_threshold = 300
    face_threshold = 1.2
    threshold = 0.5
    wight_padding = 250
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
