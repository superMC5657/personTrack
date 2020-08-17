# -*- coding: utf-8 -*-
# !@time: 2020/8/17 下午6:15
# !@author: superMC @email: 18758266469@163.com
# !@fileName: constract_facebank.py
import os
import shutil
import random


def random_choice_datasets(input_path="/home/supermc/Datasets/lfw", output_path='/home/supermc/Datasets/databank',
                           limit=2000):
    dir_list = []
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    dir_paths = os.listdir(input_path)
    if len(dir_paths) > limit:
        dir_paths = random.sample(dir_paths, k=limit)
    for dir_path in dir_paths:
        dir_path_ = os.path.join(input_path, dir_path)
        img_path = os.listdir(dir_path_)[0]
        dst_path = os.path.join(output_path, dir_path + os.path.splitext(img_path)[-1])
        img_path = os.path.join(dir_path_, img_path)
        dir_list.append(img_path)
        shutil.copy(img_path, dst_path)
    return dir_list


if __name__ == '__main__':
    random_choice_datasets()
