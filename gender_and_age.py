import cv2
import os
import numpy as np
from dnn_model import DnnModel


class GenderAndAge(DnnModel):
    def __init__(self, weights_path):
        self.hyperparameters = {'conf_threshold': 0.15}
        path_slash = weights_path.strip()
        if path_slash[-1:] != "/":
            path_slash = path_slash + "/"
        self.face_proto = path_slash + "opencv_face_detector.pbtxt"
        self.face_model = path_slash + "opencv_face_detector_uint8.pb"
        self.gender_proto = path_slash + "gender_deploy.prototxt"
        self.gender_model = path_slash + "gender_net.caffemodel"
        self.face_net = cv2.dnn.readNetFromTensorflow(self.face_model, self.face_proto)
        self.gender_net = cv2.dnn.readNet(self.gender_model, self.gender_proto)
        self.gender_model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)  # Where are these from?
        # They say that a mean = [104, 117, 123] is a standard and doesn't need to be changed nor calculated
        self.face_model_mean_values = [104, 117, 123]
        self.padding = 20  # Where are this from?
        self.gender_list = ['m', 'f']

    @staticmethod
    def __save_blob(blob_local, full_path_png):
        blob_to_save = blob_local.reshape(blob_local.shape[2] * blob_local.shape[1], blob_local.shape[3], 1)
        print(blob_to_save.shape)
        cv2.imwrite(full_path_png, blob_to_save, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def __highlight_face(self, the_frame):
        """
        This function is used only to detect faces (not gender)
        """
        frame_with_drown_squares = the_frame.copy()
        frame_height = frame_with_drown_squares.shape[0]
        frame_width = frame_with_drown_squares.shape[1]
        conf_threshold = self.hyperparameters['conf_threshold']
        blob_local = cv2.dnn.blobFromImage(frame_with_drown_squares, 1.0, (300, 300), self.face_model_mean_values, True,
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
            if confidence > conf_threshold and x1 <= frame_width and x2 <= frame_width \
                    and y1 <= frame_height and y2 <= frame_height:
                face_boxes_to_return.append([x1, y1, x2, y2])
                cv2.rectangle(frame_with_drown_squares, (x1, y1), (x2, y2), (0, 255, 0),
                              int(round(frame_height / 150)), 8)
        return frame_with_drown_squares, face_boxes_to_return

    def detect_single_frame(self, frame, a_name=None):
        result_img, face_boxes = self.__highlight_face(frame)
        height = frame.shape[0]
        width = frame.shape[1]
        if not face_boxes:
            return
        for idx, face_box in enumerate(face_boxes):
            x_from = max(0, face_box[1] - self.padding)
            x_to = min(face_box[3] + self.padding, height - 1)
            y_from = max(0, face_box[0] - self.padding)
            y_to = min(face_box[2] + self.padding, width - 1)
            face = frame[x_from:x_to, y_from:y_to]
            if idx == 0:
                cv2.imwrite(f'/Users/aneira/noticias/data/{a_name}_first_face.png', face,
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.gender_model_mean_values, swapRB=False)
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            print(f"Gender={gender}")
            if a_name is not None:
                cv2.putText(result_img, f'{gender}', (face_box[0], face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(f'/Users/aneira/noticias/data/{a_name}_processed.png', result_img,
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def detect_for_colab(self, frame):
        result_img, face_boxes = self.__highlight_face(frame)
        if not face_boxes:
            return '0f-0m', frame
        f = 0
        m = 0
        height = frame.shape[0]
        width = frame.shape[1]
        for idx, face_box in enumerate(face_boxes):
            x_from = max(0, face_box[1] - self.padding)
            x_to = min(face_box[3] + self.padding, height - 1)
            y_from = max(0, face_box[0] - self.padding)
            y_to = min(face_box[2] + self.padding, width - 1)
            face = frame[x_from:x_to, y_from:y_to]
            if idx == 0:
                cv2.imwrite(f'face_{np.random.randint(1000000)}.png', face, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.gender_model_mean_values, swapRB=False)
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            if gender == 'f':
                f = f + 1
            elif gender == 'm':
                m = m + 1

            cv2.putText(result_img, f'{gender}', (face_box[0], face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)

        return str(f) + "f-" + str(m) + "m", result_img

    def detect_single_image(self, img_path):
        basename = os.path.basename(img_path)
        file_name, _ = os.path.splitext(basename)
        frame = cv2.imread(img_path)
        return self.detect_single_frame(frame, file_name)

    def detect_single_frame_counting(self, frame, a_name=None):
        result_img, face_boxes = self.__highlight_face(frame)
        if not face_boxes:
            return '0f-0m'
        f = 0
        m = 0
        for face_box in face_boxes:
            face = frame[max(0, face_box[1] - self.padding):
                         min(face_box[3] + self.padding, frame.shape[0] - 1), max(0, face_box[0] - self.padding)
                                                                              :min(face_box[2] + self.padding,
                                                                                   frame.shape[1] - 1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.gender_model_mean_values, swapRB=False)
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            if gender == 'f':
                f = f + 1
            elif gender == 'm':
                m = m + 1

            if a_name is not None:
                cv2.putText(result_img, f'{gender}', (face_box[0], face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(f'/Users/aneira/noticias/data/{a_name}_processed.png', result_img,
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return str(f) + "f-" + str(m) + "m"

    def set_hyperparameter(self, key, value):
        self.hyperparameters[key] = value
