# -*- coding: utf-8 -*-
# !@time: 2020/7/30 下午7:58
# !@author: superMC @email: 18758266469@163.com
# !@fileName: generate_facenet_datasets.py

import os

os.environ['GLOG_minloglevel'] = '3'
import argparse
from fid.retinaFace.detector import Detector as RetinaFace
import sys
import random
import cv2
from tqdm import tqdm
import numpy as np


def main(args):
    faceDetector = RetinaFace()
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset = get_dataset(args.input_dir)
    nrof_images_total = 0
    nrof_successfully_aligned = 0

    if args.random_order:
        random.shuffle(dataset)
    for cls in tqdm(dataset):
        output_class_dir = os.path.join(output_dir, cls.name)

        if args.random_order:
            random.shuffle(cls.image_paths)

        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]  # 文件名 (不包括扩展名)
            output_filename = os.path.join(output_class_dir, filename + '.png')
            image = cv2.imread(image_path)
            if not os.path.exists(output_filename):
                faces, _ = faceDetector.forward_for_makecsv(image)
                num = len(faces)
                if num == 1:
                    if not os.path.exists(output_class_dir):
                        os.makedirs(output_class_dir)
                    face = faces[0]
                    face = face.astype(np.uint8)
                    # face = diff_resolution(face) 训练的时候再做吧.
                    face = cv2.resize(face, (args.image_size, args.image_size))
                    cv2.imwrite(output_filename, face)
                    nrof_successfully_aligned += 1
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with raw images.')
    parser.add_argument('output_dir', type=str, help='Directory with face thumbnails.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--margin', type=float,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=0.0)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')
    parser.add_argument('--gray', help='make image rgb to grey3', action='store_true')
    return parser.parse_args(argv)


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_dataset(path):
    dataset = []
    classes = os.listdir(path)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path, class_name)
        if os.path.isdir(facedir):
            images = os.listdir(facedir)
            image_paths = [os.path.join(facedir, img) for img in images]
            # 每个人对应一个class,name = 人名,paths 对应该人物所有图片路径
            dataset.append(ImageClass(class_name, image_paths))
    return dataset


def diff_resolution(image):
    generated_num = random.randint(0, 3)
    if generated_num < 1:
        image = cv2.resize(image, (28, 28))
    elif 1 <= generated_num < 3:
        image = cv2.resize(image, (56, 56))
    return image


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
