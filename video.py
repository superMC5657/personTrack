# -*- coding: utf-8 -*-
# !@time: 2020/6/10 下午10:30
# !@author: superMC @email: 18758266469@163.com
# !@fileName: video.py

import cv2

from inference import make_model, detect_person, detect_face, get_faceFeatures
from self_utils.assign_face2person import generate_person
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

        ret, frame = cap.read()
        person_images, person_boxes = detect_person(yolo, frame)
        if person_boxes:
            person_features = reid(person_images).cpu().detach()
            face_images, face_boxes = detect_face(mtcnn_detector, frame)
            if face_boxes:
                face_features = get_faceFeatures(mobileFace, face_images).cpu().detach().numpy()
                cur_person_dict = generate_person(person_features, person_boxes, face_features, face_boxes)
            else:
                cur_person_dict = generate_person(person_features, person_boxes)
            person_cache, cur_person_dict, index = update_person(index, person_cache, cur_person_dict)
            frame = plot_boxes(frame, cur_person_dict)

        # q键退出
        cv2.imshow('aoa', frame)
        k = cv2.waitKey(1)
        if (k & 0xff == ord('q')):
            break

        videoWriter.write(frame)

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
