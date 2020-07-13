# -*- coding: utf-8 -*-
# !@time: 2020/7/12 下午7:21
# !@author: superMC @email: 18758266469@163.com
# !@fileName: facenet.py

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as trans

from fid.insightFace.faceModel import MobileFaceNet, l2_norm, Backbone


class FaceNet:
    def __init__(self, weight='fid/insightFace/facenet_checkpoints/model_ir_se50.pth', use_cuda=1):
        model_name = os.path.split(weight)[-1]
        if model_name == "model_ir_se50.pth":
            self.model = Backbone(50, 0.6)
            static_dict = torch.load(weight, map_location='cuda:0')
            new_state_dict = {}
            for k, v in static_dict.items():
                name = k[7:]  # remove `vgg.`，即只取vgg.0.weights的后面几位
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        elif model_name == 'model_mobilefacenet.pth':
            self.model = MobileFaceNet(512)
            self.model.load_state_dict(torch.load(weight))

        if use_cuda:
            self.model.cuda()
        self.model.eval()
        self.use_cuda = use_cuda

    def preprocess(self, image):
        image = cv2.resize(image, (112, 112))
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mirror = trans.functional.hflip(image)
        image = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])(image)
        mirror = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])(mirror)
        return [image, mirror]

    def __call__(self, input):
        if isinstance(input, list):
            images = []
            for image in input:
                if isinstance(image, np.ndarray):
                    image = self.preprocess(image)
                else:
                    raise TypeError(
                        'Type of each element must belong to [str | numpy.ndarray]'
                    )
                images.extend(image)

        elif isinstance(input, np.ndarray):
            images = self.preprocess(input)
        else:
            raise NotImplementedError
        images = torch.stack(images, dim=0)
        if self.use_cuda:
            images = images.cuda()
        normal_features = []

        features = self.model(images)
        for i in range(0, len(features), 2):
            normal_feature = (features[i] + features[i + 1])
            normal_features.append(l2_norm(normal_feature).cpu().detach().numpy())
        return normal_features
