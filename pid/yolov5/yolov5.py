# -*- coding: utf-8 -*-
# !@time: 2020/6/20 上午8:37
# !@author: superMC @email: 18758266469@163.com
# !@fileName: yolov5.py
from self_utils.image_tool import crop_box

try:
    from utils.datasets import *
    from utils.utils import *
    from models import yolo
except:
    from .utils.datasets import *
    from .utils.utils import *
    from .models import yolo


class YoloV5:
    def __init__(self, weight='pid/yolov5/yolov5_checkpoints/yolov5l_resave.pt', use_cuda=1, half=1):
        model_name = os.path.split(weight)[-1].split('.')[0][:7]
        model_cfg = os.path.join('pid/yolov5/models', model_name + '.yaml')
        device = torch.device('cuda:0' if use_cuda else 'cpu')
        model = yolo.Model(model_cfg=model_cfg).to(device)
        model.load_state_dict(torch.load(weight))
        model.eval()
        if half:
            model.half()
        self.model = model
        self.use_cuda = use_cuda
        self.half = half

    def image_normal(self, img, image_size):
        img = letterbox(img, new_shape=image_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img

    def __call__(self, image):
        img = self.image_normal(image, image_size=640)
        # image 为numpy 格式的原始图片
        img = torch.from_numpy(img)
        if self.use_cuda:
            img = img.cuda()
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # single image
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=False)[0]
        # Apply NMS
        # predict [bbox,conf,classes]
        pred = non_max_suppression(pred, 0.6, 0.5,
                                   fast=True)[0]
        if pred is None:
            return [], []
        transform_det = scale_coords(img.shape[2:], pred[:, :4], image.shape).round()
        transform_det = transform_det.cpu().detach().numpy()
        transform_det = transform_det.astype(int).tolist()
        person_images = []
        for xyxy in transform_det:
            person_images.append(crop_box(image, xyxy))

        return person_images, transform_det


if __name__ == '__main__':
    model = YoloV5()
    image = cv2.imread('data/aoa.jpg')
    person_images, boxes = model(image)
    for index, person_image in enumerate(person_images):
        cv2.imshow('demo', person_image)
        cv2.waitKey(0)
