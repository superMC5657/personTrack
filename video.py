# -*- coding: utf-8 -*-
# !@time: 2020/6/10 下午10:30
# !@author: superMC @email: 18758266469@163.com
# !@fileName: video.py

import cv2

from inference import make_model, detect_person, get_faces, get_faceFeatures
from person import Person
from self_utils.compare import update_person
from self_utils.image_tool import plot_boxes


def main():
    yolo, reid, mtcnn_detector, mobileFace = make_model()
    person_cache = []
    cap = cv2.VideoCapture('/home/supermc/Downloads/aoa.flv')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    videoWriter = cv2.VideoWriter(
        "data/output.avi",
        cv2.VideoWriter_fourcc(*'MJPG'),  # 编码器
        fps,
        size
    )
    index = 0
    while (cap.isOpened()):
        cur_person_dict = []
        ret, frame = cap.read()
        person_images, boxes = detect_person(yolo, frame)
        for person_image, box in zip(person_images, boxes):
            person = Person()
            person.box = box
            person.pid = reid(person_image).cpu().detach()
            face = get_faces(mtcnn_detector, person_image)
            if len(face) == 1:
                person.fid = get_faceFeatures(mobileFace, face[0]).cpu().detach().numpy()
                person.findOut_name()
                '''去cache里找'''
            cur_person_dict.append(person)
        person_cache, cur_person_dict, index = update_person(index, person_cache, cur_person_dict)
        print(index)
        image = plot_boxes(frame, cur_person_dict)
        # q键退出
        cv2.imshow('aoa', image)
        k = cv2.waitKey(1)
        if (k & 0xff == ord('q')):
            break

        videoWriter.write(image)

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
