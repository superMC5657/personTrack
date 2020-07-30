# -*- coding: utf-8 -*-
# !@time: 2020/7/12 下午7:43
# !@author: superMC @email: 18758266469@163.com
# !@fileName: retinaFace.py
import cv2
import torch
import os

from torchvision.ops import nms
import numpy as np
from self_utils.image_tool import crop_box, change_coord, warp_affine
from self_utils.utils import tonumpy
from .data import cfg_mnet, cfg_re50
from .layers.functions.prior_box import PriorBox
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_gpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_gpu:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    else:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class Detector:
    # image_size = height,width
    def __init__(self, weight="fid/retinaFace/retinaFace_checkpoints/mobilenet0.25_Final.pth", image_size=(480, 640),
                 use_cuda=1):
        network = os.path.split(weight)[-1].split("_")[0]
        device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device
        self.cfg = None
        if network == "mobilenet0.25":
            self.cfg = cfg_mnet
        elif network == "Resnet50":
            self.cfg = cfg_re50
        net = RetinaFace(cfg=self.cfg, phase='test')
        net = load_model(net, weight, use_cuda)
        net.eval()
        net = net.to(device)
        self.net = net

        priorbox = PriorBox(self.cfg, image_size=image_size)
        priors = priorbox.forward()
        priors = priors.to(device)
        self.prior_data = priors.data

        scale_box = torch.Tensor([image_size[1], image_size[0], image_size[1], image_size[0]])
        self.scale_box = scale_box.to(device)

        scale_landms = torch.Tensor([image_size[1], image_size[0], image_size[1], image_size[0],
                                     image_size[1], image_size[0], image_size[1], image_size[0],
                                     image_size[1], image_size[0]])
        self.scale_landms = scale_landms.to(device)
        self.confidence_threshold = 0.5
        self.top_k = 2000
        self.keep_top_k = 150
        self.nms_threshold = 0.5
        self.image_size = image_size
        del net, priorbox, priors

    def nonMaximumSuppression(self, boxes, landms, scores):
        inds = torch.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort(descending=True)[:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS

        keep = nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        boxes = boxes[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]
        boxes = tonumpy(boxes)
        landms = tonumpy(landms)
        return boxes, landms

    def __call__(self, image):
        im_height, im_width, _ = image.shape
        height_resize = im_height / self.image_size[0]
        width_resize = im_width / self.image_size[1]
        # cv2 resize width,height
        img = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        img = img - (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).type(dtype=torch.float)
        img = img.to(self.device)

        loc, conf, landms = self.net(img)
        boxes = decode(loc.data.squeeze(0), self.prior_data, self.cfg['variance'])
        boxes = boxes * self.scale_box
        boxes = boxes.cpu()
        scores = conf.squeeze(0).data.cpu()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.cfg['variance'])
        landms = landms * self.scale_landms
        landms = landms.cpu()

        boxes, landms = self.nonMaximumSuppression(boxes, landms, scores)

        boxes[:, slice(0, 4, 2)] *= width_resize
        boxes[:, slice(1, 4, 2)] *= height_resize
        landms[:, slice(0, 10, 2)] *= width_resize
        landms[:, slice(1, 10, 2)] *= height_resize

        boxes[:, 0] = np.maximum(boxes[:, 0], 0)
        boxes[:, 1] = np.maximum(boxes[:, 1], 0)
        boxes[:, 2] = np.minimum(boxes[:, 2], im_width)  # w
        boxes[:, 3] = np.minimum(boxes[:, 3], im_height)

        faces = list()
        for box, landm in zip(boxes, landms):
            face = crop_box(image, box)
            eye_left_x, eye_left_y = change_coord(landm[0], landm[1], box[0], box[1])
            eye_right_x, eye_right_y = change_coord(landm[2], landm[3], box[0], box[1])
            face = warp_affine(image=face, x1=eye_left_x, y1=eye_left_y, x2=eye_right_x, y2=eye_right_y)
            faces.append(face)
        return faces, boxes

    def forward_for_makecsv(self, image):

        im_height, im_width, _ = image.shape
        image_size = (im_height, im_width)
        priorbox = PriorBox(self.cfg, image_size=image_size)
        priors = priorbox.forward()
        priors = priors.to(self.device)
        self.prior_data = priors.data

        scale_box = torch.Tensor([image_size[1], image_size[0], image_size[1], image_size[0]])
        self.scale_box = scale_box.to(self.device)

        scale_landms = torch.Tensor([image_size[1], image_size[0], image_size[1], image_size[0],
                                     image_size[1], image_size[0], image_size[1], image_size[0],
                                     image_size[1], image_size[0]])
        self.scale_landms = scale_landms.to(self.device)

        # cv2 resize width,height
        img = image - (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).type(dtype=torch.float)
        img = img.to(self.device)

        loc, conf, landms = self.net(img)
        boxes = decode(loc.data.squeeze(0), self.prior_data, self.cfg['variance'])
        boxes = boxes * self.scale_box
        boxes = boxes.cpu()
        scores = conf.squeeze(0).data.cpu()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.cfg['variance'])
        landms = landms * self.scale_landms
        landms = landms.cpu()

        boxes, landms = self.nonMaximumSuppression(boxes, landms, scores)

        boxes[:, 0] = np.maximum(boxes[:, 0], 0)
        boxes[:, 1] = np.maximum(boxes[:, 1], 0)
        boxes[:, 2] = np.minimum(boxes[:, 2], im_width)  # w
        boxes[:, 3] = np.minimum(boxes[:, 3], im_height)

        faces = list()
        for box, landm in zip(boxes, landms):
            face = crop_box(image, box)
            eye_left_x, eye_left_y = change_coord(landm[0], landm[1], box[0], box[1])
            eye_right_x, eye_right_y = change_coord(landm[2], landm[3], box[0], box[1])
            face = warp_affine(image=face, x1=eye_left_x, y1=eye_left_y, x2=eye_right_x, y2=eye_right_y)
            faces.append(face)
        return faces, boxes


if __name__ == '__main__':
    detector = Detector()
