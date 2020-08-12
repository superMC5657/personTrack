# -*- coding: utf-8 -*-
# !@time: 2020/8/12 下午1:41
# !@author: superMC @email: 18758266469@163.com
# !@fileName: rename.py
import os
import shutil

file_dir = 'data/person_with_name'
file_names = os.listdir(file_dir)
for file_name in file_names:
    shutil.move(os.path.join(file_dir, file_name), os.path.join(file_dir, file_name.lower()))
