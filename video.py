# -*- coding: utf-8 -*-
# !@time: 2020/6/10 下午10:30
# !@author: superMC @email: 18758266469@163.com
# !@fileName: video.py
import argparse
import os
import sys
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
from self_utils.callback_func import callback_progress
from self_utils.person_utils import generate_person, compression_person, update_person
from self_utils.image_tool import plot_boxes_pil
from self_utils.utils import compute_time, write_person, get_video_duration_cv2

torch.set_grad_enabled(False)


def video(src_video, dst_video, dst_txt,
          callback_progress=None, callback_video=None):
    src_video_cap = cv2.VideoCapture(src_video)
    src_video_fps = src_video_cap.get(cv2.CAP_PROP_FPS)
    video_size = (
        int(src_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(src_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    is_video = opt.is_video  # 是否为视频 (如果实时 代码逻辑会不同 需要靠time模块计时)
    video_speed = opt.video_speed  # 视频播放速度

    record_time = opt.record_time  # 计时间隔时间
    fps_record_time = src_video_fps * record_time  # 计时间隔帧数

    compress_time = opt.compress_time  # 压缩person_cache 时间
    fps_compress_time = src_video_fps * compress_time  # 压缩person_cache 帧数

    callback_time = opt.callback_time  # 回调间隔时间
    fps_callback_time = src_video_fps * callback_time  # 回调间隔帧数

    last_compress_timing = 0  # 上一次压缩person_cache的次数
    last_record_timing = 0  # 上一次计时的次数
    last_callback_timing = 0  # 上一次回调的次数

    frame_num = 0  # 记录帧数
    person_id = 0  # 记录person_id
    vis_video = opt.vis_video  # 是否可视化
    show_fps = opt.show_fps  # 是否反馈fps

    video_total_time = get_video_duration_cv2(src_video)
    if dst_video:
        videoWriter = cv2.VideoWriter(
            dst_video,
            cv2.VideoWriter_fourcc(*'MJPG'),  # 编码器
            src_video_fps / video_speed,
            (video_size[0] + opt.wight_padding, video_size[1])
        )

    yolo = YoloV5()
    reid = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='pid/deep_person_reid/checkpoints/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
        verbose=False)
    face_detector = RetinaFace(image_size=(video_size[1], video_size[0]))
    # detector = MTCNN()
    faceNet = FaceNet()
    person_caches = []
    while src_video_cap.isOpened():
        start_time = time.time()  # start time of the loop
        frame_num += 1
        ret, frame = src_video_cap.read()
        if frame_num % video_speed != 0:
            continue
        if not ret:
            break

        wight_board = np.zeros((video_size[1], opt.wight_padding, 3), dtype=np.uint8)
        image = np.concatenate((frame, wight_board), axis=1)
        person_images, person_boxes = yolo(frame)
        if len(person_boxes) > 0:
            face_features, face_boxes = None, None
            person_features = reid(person_images).cpu().detach()
            face_images, face_boxes, face_effective = face_detector(frame)
            if face_effective:
                face_features = faceNet(face_images)

            person_current = generate_person(person_features, person_boxes, face_features, face_boxes,
                                             face_effective)
            person_current, person_caches, person_id = update_person(person_id, person_current, person_caches)
            image = plot_boxes_pil(image, person_current, src_video_fps / video_speed)

        compress_timing = frame_num * video_speed // fps_compress_time
        if compress_timing - last_compress_timing >= 1:
            last_compress_timing = compress_timing
            person_caches = compression_person(person_caches)

        if is_video:
            record_timing = frame_num * video_speed // fps_record_time
            # record_timing_times 计时的次数(或许speed_step太快导致 一帧就超过了计时间隔/或者FPS太小导致一帧就超过了计时间隔)
            record_timing_times = record_timing - last_record_timing
            if record_timing_times >= 1:
                last_record_timing = record_timing
                person_caches = compute_time(person_caches, record_time * int(record_timing_times))

        if callback_progress:
            callback_timing = frame_num * video_speed // fps_callback_time
            if callback_timing - last_callback_timing >= 1:
                last_callback_timing = callback_timing
                percentage = callback_timing * callback_time / video_total_time
                callback_progress(percentage)

        if callback_video:
            callback_video(image)
        # q键退出
        if vis_video:
            cv2.imshow(os.path.split(src_video)[-1], image)
            k = cv2.waitKey(1)
            if k & 0xff == ord('q'):
                break
        if dst_video:
            videoWriter.write(image)

        if show_fps:
            if frame_num % (10 * video_speed) == 0:
                print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop
    person_caches = compression_person(person_caches)
    if dst_txt:
        write_person(person_caches, dst_txt)
    src_video_cap.release()
    if dst_video:
        videoWriter.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Video")
    parser.add_argument('-i', '--src_video', type=str, help="输入视频路径")
    parser.add_argument('-o', '--dst_video', type=str, help="输出视频路径")
    parser.add_argument('--dst_txt', type=str, help="输出文本路径")

    return parser.parse_args(argv)


def main(args):
    start_time = time.time()
    video(args.src_video, args.dst_video, args.dst_txt, callback_progress=callback_progress)
    print(time.time() - start_time)


if __name__ == '__main__':
    arg = parse_arguments(sys.argv[1:])
    main(arg)
