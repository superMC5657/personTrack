# -*- coding: utf-8 -*-
# !@time: 2020/7/13 下午2:33
# !@author: superMC @email: 18758266469@163.com
# !@fileName: mtcnn.py

from fid.mtcnn.detect import create_mtcnn_net, MtcnnDetector

from self_utils.face_utils import crop_faces


class MTCNN:
    def __init__(self, p_model_path="fid/mtcnn/mtcnn_checkpoints/pnet_epoch.pt",
                 r_model_path="fid/mtcnn/mtcnn_checkpoints/rnet_epoch.pt",
                 o_model_path="fid/mtcnn/mtcnn_checkpoints/onet_epoch.pt", use_cuda=1, min_face_size=15):
        pnet, rnet, onet = create_mtcnn_net(p_model_path, r_model_path, o_model_path, use_cuda)
        self.detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=min_face_size)

    def __call__(self, image):
        # start = time.time()
        # for i in range(100):
        #     bboxs, landmarks = detector.detect_face(image)
        # print(time.time() - start)
        boxes, landmarks = self.detector.detect_face(image)
        return crop_faces(image, boxes, landmarks)
