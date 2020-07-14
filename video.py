# -*- coding: utf-8 -*-
# !@time: 2020/6/10 下午10:30
# !@author: superMC @email: 18758266469@163.com
# !@fileName: video.py
import time

import cv2
import torch
from torch.backends import cudnn
from torchreid.utils import FeatureExtractor

from fid.insightFace.faceNet import FaceNet
from fid.mtcnn.mtcnn import MTCNN
from fid.retinaFace.detector import Detector as RetinaFace
from pid.yolov5.yolov5 import YoloV5
from self_utils.assign_face2person import generate_person
from self_utils.compare import update_person
from self_utils.image_tool import plot_boxes

cudnn.benchmark = True


def main():
    yolo = YoloV5()
    # model = yolov5_model("pid/yolov5/weights/yolov5l_resave.pt")
    reid = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='deep_person_reid/checkpoints/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
    )
    # detector = RetinaFace()
    detector = MTCNN()
    faceNet = FaceNet()
    person_cache = []
    cap = cv2.VideoCapture('data/data0.avi')
    fps = cap.get(cv2.CAP_PROP_FPS)
    speed = 1
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    videoWriter = cv2.VideoWriter(
        "data/output.avi",
        cv2.VideoWriter_fourcc(*'MJPG'),  # 编码器
        fps / speed,
        size
    )
    frame_num = 0
    index = 0
    vis = True
    while cap.isOpened():
        start_time = time.time()  # start time of the loop
        frame_num += 1
        ret, frame = cap.read()
        if frame_num % speed != 0:
            continue

        if not ret:
            break

        person_images, person_boxes = yolo(frame)
        if person_boxes:
            face_features, face_boxes = None, None
            person_features = reid(person_images).cpu().detach()
            face_images, face_boxes = detector(frame)
            if len(face_boxes) > 0:
                face_features = faceNet(face_images)
            cur_person_dict = generate_person(person_features, person_boxes, face_features, face_boxes)
            person_cache, cur_person_dict, index = update_person(index, person_cache, cur_person_dict)
            frame = plot_boxes(frame, cur_person_dict)

        # q键退出
        if vis:
            cv2.imshow('aoa', frame)
            k = cv2.waitKey(1)
            if (k & 0xff == ord('q')):
                break
        videoWriter.write(frame)
        if frame_num % (10 * speed) == 0:
            print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with torch.no_grad():
        main()
