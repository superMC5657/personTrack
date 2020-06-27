# -*- coding: utf-8 -*-
# !@time: 2020/6/10 上午10:35
# !@author: superMC @email: 18758266469@163.com
# !@fileName: inference.py

import cv2
from torchreid.utils import FeatureExtractor

from fid.inference import get_faces, get_faceFeatures
from fid.inference import mobile_face_model
from fid.mtcnn.detect import create_mtcnn_net, MtcnnDetector

# from pid.yolov4.yolov4 import yolov4_model as yolo_model
# from pid.yolov4.yolov4 import detect_person


from pid.yolov5.yolov5 import yolov5_model as yolo_model
from pid.yolov5.yolov5 import detect_person


def make_model():
    yolo = yolo_model()
    reid = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='pid/deep_person_reid/checkpoints/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
        verbose=False)
    pnet, rnet, onet = create_mtcnn_net(p_model_path="fid/mtcnn/mtcnn_checkpoints/pnet_epoch.pt",
                                        r_model_path="fid/mtcnn/mtcnn_checkpoints/rnet_epoch.pt",
                                        o_model_path="fid/mtcnn/mtcnn_checkpoints/onet_epoch.pt", use_cuda=True)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)
    mobileFace = mobile_face_model("fid/InsightFace_Pytorch/facenet_checkpoints/model_ir_se50.pth")
    return yolo, reid, mtcnn_detector, mobileFace


if __name__ == '__main__':
    image = cv2.imread("data/aoa.jpg")
    yolo, reid, mtcnn_detector, mobileFace = make_model()
    person_images, _ = detect_person(yolo, image)
    for person_image in person_images:
        faces = get_faces(mtcnn_detector, person_image)
        cv2.imshow("person", person_image)
        cv2.waitKey(0)
        print(len(faces))
        for face in faces:
            cv2.imshow("demo", face)
            cv2.waitKey()
