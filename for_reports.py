import cv2
import os
import numpy as np
from gender_and_age import GenderAndAge


class ForReports(GenderAndAge):
    def __init__(self, weights_path):
        self.hyperparameters = {'conf_threshold': 0.15, 'gender_threshold': 0.5}
        path_slash = weights_path.strip()
        if path_slash[-1:] != "/":
            path_slash = path_slash + "/"
        self.face_proto = path_slash + "opencv_face_detector.pbtxt"
        self.face_model = path_slash + "opencv_face_detector_uint8.pb"
        self.gender_proto = path_slash + "gender_deploy.prototxt"
        self.gender_model = path_slash + "gender_net.caffemodel"
        self.face_net = cv2.dnn.readNetFromTensorflow(self.face_model, self.face_proto)
        self.gender_net = cv2.dnn.readNetFromCaffe(self.gender_proto, self.gender_model)
        self.gender_model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)  # Where are these from?
        # They say that a mean = [104, 117, 123] is a standard and doesn't need to be changed nor calculated
        self.face_model_mean_values = [104, 117, 123]
        self.padding = 20  # Where is this from?
        self.gender_list = ['m', 'f']

    def __highlight_face(self, the_frame, a_name=None):
        frame_with_drown_squares = np.zeros([360, 640, 3], dtype=np.uint8)
        frame_with_drown_squares.fill(255)
        frame_height = frame_with_drown_squares.shape[0]
        frame_width = frame_with_drown_squares.shape[1]
        conf_threshold = self.hyperparameters['conf_threshold']
        blob_local = cv2.dnn.blobFromImage(the_frame, 1.0, (300, 300), self.face_model_mean_values, True,
                                           False)
        self.face_net.setInput(blob_local)
        detections = self.face_net.forward()
        face_boxes_to_return = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            if confidence > conf_threshold and x1 <= frame_width and x2 <= frame_width and y1 <= frame_height and y2 <= frame_height:
                face_boxes_to_return.append([x1, y1, x2, y2])
                cv2.rectangle(frame_with_drown_squares, (x1, y1), (x2, y2), (0, 255, 0),
                              int(round(frame_height / 150)), 8)
        return frame_with_drown_squares, face_boxes_to_return

    def detect_single_frame_blank(self, frame, a_name=None):
        result_img, face_boxes = self.__highlight_face(frame, a_name)
        height = frame.shape[0]
        width = frame.shape[1]
        return result_img
