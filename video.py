# -*- coding: utf-8 -*-
# !@time: 2020/6/10 下午10:30
# !@author: superMC @email: 18758266469@163.com
# !@fileName: video.py
import time
import numpy as np
import cv2
import torch
from torch.backends import cudnn
from torchreid.utils import FeatureExtractor

from config import opt
from fid.insightFace.faceNet import FaceNet
from fid.mtcnn.mtcnn import MTCNN
from fid.retinaFace.detector import Detector as RetinaFace
from pid.yolov5.yolov5 import YoloV5
from self_utils.person_utils import generate_person, compression_person, update_person
from self_utils.image_tool import plot_boxes, plot_boxes_pil
from self_utils.utils import compute_time

cudnn.benchmark = True
torch.set_grad_enabled(False)


def main():
    person_cache = []
    cap = cv2.VideoCapture('data/data1.avi')
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_speed = opt.video_speed  # 视频播放速度
    record_time = opt.record_time  # 计时间隔时间
    fps_record_time = fps * record_time  # 计时间隔帧数
    is_video = opt.is_video  # 是否为视频 (如果实时 代码逻辑会不同 需要靠time模块计时)

    time_flag = 0  # 判断是否计时的条件

    compress_time = opt.compress_time  # 压缩person_cache 时间
    fps_compress_time = fps * compress_time  # 压缩person_cache 帧数

    last_compress_timing = 0
    last_record_timing = 0

    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    videoWriter = cv2.VideoWriter(
        "data/output.avi",
        cv2.VideoWriter_fourcc(*'MJPG'),  # 编码器
        fps / video_speed,
        (size[0] + opt.wight_padding, size[1])
    )
    frame_num = 0  # 记录帧数
    person_id = 0  # 记录person_id
    vis = opt.vis  # 是否可视化
    yolo = YoloV5()
    reid = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='pid/deep_person_reid/checkpoints/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
        verbose=False)
    detector = RetinaFace(image_size=(size[1], size[0]))
    # detector = MTCNN()
    faceNet = FaceNet()
    while cap.isOpened():
        start_time = time.time()  # start time of the loop
        frame_num += 1
        ret, frame = cap.read()
        time_flag += video_speed  # 根据speed_step 记录
        if frame_num % video_speed != 0:
            continue
        if not ret:
            break

        wight_board = np.zeros((size[1], opt.wight_padding, 3), dtype=np.uint8)
        image = np.concatenate((frame, wight_board), axis=1)
        person_images, person_boxes = yolo(frame)
        if person_boxes:
            face_features, face_boxes = None, None
            person_features = reid(person_images).cpu().detach()
            face_images, face_boxes = detector(frame)
            if len(face_boxes) > 0:
                face_features = faceNet(face_images)
            person_current = generate_person(person_features, person_boxes, face_features, face_boxes)
            person_current, person_cache, person_id = update_person(person_id, person_current, person_cache)
            image = plot_boxes_pil(image, person_current, fps / video_speed)

        compress_timing = time_flag / fps_compress_time
        if compress_timing - last_compress_timing >= 1:
            last_compress_timing = compress_timing
            person_cache = compression_person(person_cache)

        if is_video:
            record_timing = time_flag / fps_record_time
            # record_timing_times 计时的次数(或许speed_step太快导致 一帧就超过了计时间隔/或者FPS太小导致一帧就超过了计时间隔)
            record_timing_times = record_timing - last_record_timing
            if record_timing_times >= 1:
                last_record_timing = record_timing
                person_cache = compute_time(person_cache, record_time * int(record_timing_times))

        # q键退出
        if vis:
            cv2.imshow('frame', image)
            k = cv2.waitKey(1)
            if k & 0xff == ord('q'):
                break
        videoWriter.write(image)
        if frame_num % (10 * video_speed) == 0:
            print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop
    person_cache = compression_person(person_cache)
    write_person(person_cache, fps / video_speed)
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


def write_person(person_cache, fps):
    file = open('data/output.txt', "w", encoding='utf-8')

    for person in person_cache:
        line = str(person.id) + "\t" + person.name + '\t' + str(person.time) + "\n"
        file.write(line)
    file.close()


if __name__ == '__main__':
    main()
