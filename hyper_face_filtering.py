import cv2
import os
import numpy as np
from dnn_model import DnnModel
import chainer
from hyperface.scripts.models import HyperFaceModel
from hyperface.scripts.models import IMG_SIZE


class HyperFaceFiltering(DnnModel):
    def __init__(self, opencv_pbtxt_path, opencv_pb_path, hyperface_path):
        os.environ["CHAINER_TYPE_CHECK"] = "0"
        self.face_proto = opencv_pbtxt_path
        self.face_model = opencv_pb_path
        self.face_net = cv2.dnn.readNetFromTensorflow(self.face_model, self.face_proto)
        self.padding = 0
        self.gender_list = ['m', 'f']
        self.hyperface_threshold = 0.5
        self.gaa_threshold = 0.5
        self.hyperface_model = HyperFaceModel()
        self.hyperface_model.train = False
        self.hyperface_model.report = False
        self.hyperface_model.backward = False
        self.hyperface_model_path = hyperface_path
        self.face_model_mean_values = [104, 117, 123]
        # Initialize model
        chainer.serializers.load_npz(self.hyperface_model_path, self.hyperface_model)

    @staticmethod
    def _cvt_variable(v):
        # Convert from chainer variable
        if isinstance(v, chainer.variable.Variable):
            v = v.data
            if hasattr(v, 'get'):
                v = v.get()
        return v

    @staticmethod
    def __save_blob(blob_local, full_path_png):
        blob_to_save = blob_local.reshape(blob_local.shape[2] * blob_local.shape[1], blob_local.shape[3], 1)
        print(blob_to_save.shape)
        cv2.imwrite(full_path_png, blob_to_save, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def __get_all_bounding_boxes(self, the_frame):
        frame_with_drown_squares = the_frame.copy()
        frame_height = frame_with_drown_squares.shape[0]
        frame_width = frame_with_drown_squares.shape[1]
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
            face_boxes_to_return.append([x1, y1, x2, y2, confidence])
        return face_boxes_to_return

    def get_all_bounding_boxes(self, frame, lower, upper):
        result_img = frame.copy()
        face_boxes = self.__get_all_bounding_boxes(result_img)
        if not face_boxes:
            return 0, frame
        n_frames = 0
        for idx, face_box in enumerate(face_boxes):
            x1 = face_box[0]
            y1 = face_box[1]
            x2 = face_box[2]
            y2 = face_box[3]
            area = abs(x2 - x1) * abs(y2 - y1)
            if area > upper or area < lower:
                continue
            n_frames = n_frames + 1
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 1, 8)
            cv2.putText(result_img, f'{area}', (x1, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        color, 1, cv2.LINE_AA)
        return n_frames, result_img

    def detect_single_image(self, img_path):
        pass

    def detect_single_frame(self, frame, a_name=None):
        result_img = frame.copy()
        face_boxes = self.__get_all_bounding_boxes(result_img)
        if not face_boxes:
            return '0f-0m', frame
        height = frame.shape[0]
        width = frame.shape[1]
        f = 0
        m = 0
        for idx, face_box in enumerate(face_boxes):
            x_from = max(0, face_box[1] - self.padding)
            x_to = min(face_box[3] + self.padding, height - 1)
            y_from = max(0, face_box[0] - self.padding)
            y_to = min(face_box[2] + self.padding, width - 1)
            if x_from >= x_to or y_from >= y_to:
                continue
            face = frame[x_from:x_to, y_from:y_to]
            p_detection, p_gender = self.is_face(face)
            cv2.imwrite(f'/Users/aneira/noticias/data/det_{str(round(face_box[4] * 100.0, 2))}_{str(round(p_detection * 100.0, 2))}.png', face,
                        [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if p_detection < self.hyperface_threshold:
                continue
            gender = 'f'
            if p_gender <= 0.5:
                gender = 'm'
            if gender == 'f':
                f = f + 1
            elif gender == 'm':
                m = m + 1
            cv2.rectangle(result_img, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 255, 0),
                          int(round(height / 150)), 8)
            # cv2.putText(result_img, f'{gender.upper()}', (face_box[0], face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            #             (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_img, str(round(face_box[4] * 100.0, 2)), (face_box[0], face_box[3]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (0, 255, 255), 1, cv2.LINE_AA)
        return str(f) + "f-" + str(m) + "m", result_img

    def is_face(self, img):

        xp = np

        img = img.astype(np.float32) / 255.0  # [0:1]
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX)
        img = np.transpose(img, (2, 0, 1))

        # Create single batch
        imgs = xp.asarray([img])
        x = chainer.Variable(imgs)  # , volatile=True)

        # Forward
        y = self.hyperface_model(x)

        # Chainer.Variable -> np.ndarray
        detection = self._cvt_variable(y['detection'])
        genders = self._cvt_variable(y['gender'])

        gender = genders[0]

        return detection[0], gender

    def detect_for_colab(self, frame, a_name):
        pass
