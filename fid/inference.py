# -*- coding: utf-8 -*-
# !@time: 2020/6/10 上午9:27
# !@author: superMC @email: 18758266469@163.com
# !@fileName: face_model.py
import os
from PIL import Image
from fid.mtcnn.detect import create_mtcnn_net, MtcnnDetector
from self_utils.image_tool import change_coord, warp_affine, crop_box
from torchvision import transforms as trans
from fid.InsightFace_Pytorch.face_model import MobileFaceNet, l2_norm, Backbone
import cv2
import torch
import numpy as np


def mobile_face_model(weight='fid/InsightFace_Pytorch/facenet_checkpoints/model_ir_se50.pth', use_cuda=1):
    model_name = os.path.split(weight)[-1]
    if model_name == "model_ir_se50.pth":
        model = Backbone(50, 0.6)
        static_dict = torch.load(weight, map_location='cuda:0')
        new_state_dict = {}
        for k, v in static_dict.items():
            name = k[7:]  # remove `vgg.`，即只取vgg.0.weights的后面几位
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    elif model_name == 'model_mobilefacenet.pth':
        model = MobileFaceNet(512)
        model.load_state_dict(torch.load(weight))

    if use_cuda:
        model.cuda()
    model.eval()
    return model


def detect_face(detector, image):
    # start = time.time()
    # for i in range(100):
    #     bboxs, landmarks = detector.detect_face(image)
    # print(time.time() - start)
    bboxs, landmarks = detector.detect_face(image)
    width = image.shape[1]
    height = image.shape[0]
    face_boxes = []
    faces = []
    for box, landmark in zip(bboxs, landmarks):
        box[0] = np.maximum(box[0], 0)
        box[1] = np.maximum(box[1], 0)
        box[2] = np.minimum(box[2], width)  # w
        box[3] = np.minimum(box[3], height)

        face = crop_box(image, box)
        eye_left_x, eye_left_y = change_coord(landmark[0], landmark[1], box[0], box[1])
        eye_right_x, eye_right_y = change_coord(landmark[2], landmark[3], box[0], box[1])
        try:
            face = warp_affine(image=face, x1=eye_left_x, y1=eye_left_y, x2=eye_right_x,
                               y2=eye_right_y)
        except:
            print(box, landmark)
            cv2.imwrite('data/error.png', image)
        faces.append(face)
        face_boxes.append([int(_) for _ in box[:4]])
    return faces, face_boxes


def preprocess(image):
    image = cv2.resize(image, (112, 112))
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 均值为0.5 标准差为 0.5的分布
    image = trans.Compose([
        trans.ToTensor(),  # 0-1
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])(image)
    return image


def get_faceFeatures(faceModel, input, use_cuda=1):
    if isinstance(input, list):
        images = []
        for image in input:
            if isinstance(image, np.ndarray):
                image = preprocess(image)
            else:
                raise TypeError(
                    'Type of each element must belong to [str | numpy.ndarray]'
                )
            images.append(image)
        images = torch.stack(images, dim=0)

    elif isinstance(input, np.ndarray):
        image = preprocess(input)
        images = image.unsqueeze(0)
    else:
        raise NotImplementedError
    if use_cuda:
        images = images.cuda()
    with torch.no_grad():
        features = faceModel(images)

    return features


if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(p_model_path="fid/mtcnn/mtcnn_checkpoints/pnet_epoch.pt",
                                        r_model_path="fid/mtcnn/mtcnn_checkpoints/rnet_epoch.pt",
                                        o_model_path="fid/mtcnn/mtcnn_checkpoints/onet_epoch.pt", use_cuda=True)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)
    faceModel = mobile_face_model()
    img = cv2.imread("data/aoa.jpg")
    faces, _ = detect_face(mtcnn_detector, img)
    for id, face in enumerate(faces):
        image_path = os.path.join("data/face_with_name", chr(id + 65) + ".png")
        cv2.imwrite(image_path, face)
        cv2.imshow('demo', face)
        cv2.waitKey(0)
    features = get_faceFeatures(faceModel, faces)
    print(features.shape)
