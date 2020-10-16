# -*- coding: utf-8 -*-
# !@time: 2020/6/10 下午10:30
# !@author: superMC @email: 18758266469@163.com
# !@fileName: demo.py
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from PyQt5.QtCore import QThread
from torchreid.utils import FeatureExtractor

from config import opt
from fid.insightFace.faceNet import FaceNet
from fid.retinaFace.detector import Detector as RetinaFace
from pid.yolov5.yolov5 import YoloV5
from self_utils.image_tool import plot_boxes_pil
from self_utils.models import init_models
from self_utils.person_tracker import generate_person, update_person
from self_utils.person_utils import compression_person
from self_utils.utils import compute_time, write_person, get_video_duration_cv2

torch.set_grad_enabled(False)


class Demo(QThread):
    def __init__(self, src_video, dst_video, dst_txt, models, callback_progress=None, callback_video=None,
                 is_video=False):
        # 是否为视频 (如果实时 代码逻辑会不同 需要靠time模块计时)
        super().__init__()
        self.src_video = src_video
        self.dst_video = dst_video
        self.dst_txt = dst_txt
        self.callback_progress = callback_progress
        self.callback_video = callback_video
        self.is_video = is_video
        self.models = models

    def run(self):

        src_video_cap = cv2.VideoCapture(self.src_video)
        src_video_fps = src_video_cap.get(cv2.CAP_PROP_FPS)
        video_size = (
            int(src_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(src_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

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
        is_break = False

        if video_speed >= src_video_fps * min(record_time, callback_time, compress_time):
            print("fatal error")
        if self.is_video:
            video_total_time = get_video_duration_cv2(self.src_video)
        if self.dst_video:
            videoWriter = cv2.VideoWriter(
                self.dst_video,
                cv2.VideoWriter_fourcc(*'MJPG'),  # 编码器
                src_video_fps / video_speed,
                (video_size[0] + opt.wight_padding, video_size[1])
            )

        person_caches = []
        yolo, reid, face_detector, faceNet = self.models
        start_time = time.time()
        fps_time_t_1 = start_time
        while src_video_cap.isOpened():

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
                image = plot_boxes_pil(image, person_current)

            if self.is_video:
                record_timing = frame_num // fps_record_time
                compress_timing = frame_num // fps_compress_time

            else:
                time_stride = time.time() - start_time
                compress_timing = time_stride // compress_time
                record_timing = time_stride // record_time

            if compress_timing - last_compress_timing >= 1:
                last_compress_timing = compress_timing
                person_caches = compression_person(person_caches)

            if record_timing - last_record_timing >= 1:
                last_record_timing = record_timing
                person_caches = compute_time(person_caches, record_time)

            if self.is_video:
                if self.callback_progress:
                    callback_timing = frame_num // fps_callback_time
                    if callback_timing - last_callback_timing >= 1:
                        last_callback_timing = callback_timing
                        percentage = callback_timing * callback_time / video_total_time
                        self.callback_progress(percentage)

            if self.callback_video:
                is_break = self.callback_video(image)

            # q键退出
            if vis_video:
                cv2.imshow(os.path.split(self.src_video)[-1], image)
                k = cv2.waitKey(1)
                if k & 0xff == ord('q'):
                    break
            if self.dst_video:
                videoWriter.write(image)

            if show_fps:
                if frame_num % (show_fps * video_speed) == 0:
                    fps_time_t = time.time()
                    print("FPS: ", show_fps / (fps_time_t - fps_time_t_1))  # FPS = 1 / time to process loop
                    fps_time_t_1 = fps_time_t
            if is_break:
                break
            ret, image = cv2.imencode('.jpg', image)
            if self.is_video:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image.tobytes() + b'\r\n\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image.tobytes() + b'\r\n\r\n')

        person_caches = compression_person(person_caches)
        if self.dst_txt:
            write_person(person_caches, self.dst_txt)
        src_video_cap.release()
        if self.dst_video:
            videoWriter.release()
        cv2.destroyAllWindows()
        return None


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument('-i', '--src_video', type=str, help="输入视频路径")
    parser.add_argument('-o', '--dst_video', type=str, help="输出视频路径")
    parser.add_argument('--dst_txt', type=str, help="输出文本路径")

    return parser.parse_args(argv)


def main(args):
    start_time = time.time()
    models = init_models()
    demo = Demo(args.src_video, args.dst_video, args.dst_txt, models)
    demo.run()
    print(time.time() - start_time)


if __name__ == '__main__':
    arg = parse_arguments(sys.argv[1:])
    main(arg)
