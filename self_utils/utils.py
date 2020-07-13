# -*- coding: utf-8 -*-
# !@time: 2020/6/28 下午4:05
# !@author: superMC @email: 18758266469@163.com
# !@fileName: utils.py
import pandas as pd
import numpy as np
import torch


def get_data(csv_path):
    name_features_dataframe = pd.read_csv(csv_path, sep=',')
    name_dataframe = name_features_dataframe[['Name']]
    features_name = ['Features%d' % i for i in range(512)]
    features_dataframe = name_features_dataframe[features_name]
    labels = name_dataframe.values
    features = features_dataframe.values
    labels = np.squeeze(labels).tolist()
    return labels, features


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
