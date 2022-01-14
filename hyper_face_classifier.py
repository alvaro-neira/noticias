import random
import cv2
import os
import numpy as np
from dnn_model import DnnModel
import chainer
from hyperface.scripts import models
from hyperface.scripts.drawing import draw_gender, draw_detection_in_orig


class HyperFaceClassifier(DnnModel):
    def __init__(self, face_model, face_proto, hyperface_pre_trained, height, width, random_seed):
        random.seed(random_seed)
        self.hyperparameters = {'conf_threshold': 0.5, 'gender_threshold': 0.5}
        self.face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
        # They say that a mean = [104, 117, 123] is a standard and doesn't need to be changed nor calculated
        self.face_model_mean_values = [104, 117, 123]
        self.padding = 20  # Where is this from?
        self.gender_list = ['m', 'f']
        self.frame_height = height
        self.frame_width = width

        # Disable type check in chainer
        os.environ["CHAINER_TYPE_CHECK"] = "0"
        # Define a model
        self.hyperface_model = models.HyperFaceModel()
        self.hyperface_model.train = False
        self.hyperface_model.report = False
        self.hyperface_model.backward = False
        # Initialize model
        chainer.serializers.load_npz(hyperface_pre_trained, self.hyperface_model)

    @staticmethod
    def _cvt_variable(v):
        # Convert from chainer variable
        if isinstance(v, chainer.variable.Variable):
            v = v.data
            if hasattr(v, 'get'):
                v = v.get()
        return v

    def get_gender_and_draw(self, img, final_scene, face_box):

        frame = img.copy()
        frame = frame.astype(np.float32) / 255.0  # [0:1]
        frame = cv2.resize(frame, models.IMG_SIZE)
        frame = cv2.normalize(frame, None, -0.5, 0.5, cv2.NORM_MINMAX)
        frame = np.transpose(frame, (2, 0, 1))

        # Create single batch
        imgs = np.asarray([frame])
        x = chainer.Variable(imgs)  # , volatile=True)

        y = self.hyperface_model(x)

        # Chainer.Variable -> np.ndarray
        imgs = HyperFaceClassifier._cvt_variable(y['img'])
        detections = HyperFaceClassifier._cvt_variable(y['detection'])
        landmarks = HyperFaceClassifier._cvt_variable(y['landmark'])
        visibilities = HyperFaceClassifier._cvt_variable(y['visibility'])
        poses = HyperFaceClassifier._cvt_variable(y['pose'])
        genders = HyperFaceClassifier._cvt_variable(y['gender'])

        # Use first data in one batch
        frame = imgs[0]
        detection = detections[0]
        landmark = landmarks[0]
        visibility = visibilities[0]
        pose = poses[0]
        gender = genders[0]

        frame = np.transpose(frame, (1, 2, 0))
        frame = frame.copy()
        frame += 0.5  # [-0.5:0.5] -> [0:1]
        detection = (detection > 0.5)
        gender = (gender > self.hyperparameters['gender_threshold'])

        # Draw results
        total_height = final_scene.shape[0]
        total_width = final_scene.shape[1]
        x1 = face_box[0]
        y1 = face_box[1]
        x2 = face_box[2]
        y2 = face_box[3]
        height = y2 - y1
        width = x2 - x1
        scale = 2 * width / total_width

        draw_detection_in_orig(detection, final_scene, face_box, scale)
        # landmark_color = (0, 1, 0) if detection == 1 else (0, 0, 1)
        # draw_landmark(frame, landmark, visibility, landmark_color, 0.5)
        # draw_pose(frame, pose)
        draw_gender(gender, final_scene, face_box, scale)
        # return 255 * cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        return gender

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
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes_to_return.append([x1, y1, x2, y2])
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

    def __highlight_face(self, the_frame, a_name=None):
        """
        This function is used only to detect faces (not gender)
        """
        frame_with_drown_squares = the_frame.copy()

        conf_threshold = self.hyperparameters['conf_threshold']
        blob_local = cv2.dnn.blobFromImage(frame_with_drown_squares, 1.0, (300, 300), self.face_model_mean_values, True,
                                           False)
        self.face_net.setInput(blob_local)
        detections = self.face_net.forward()
        face_boxes_to_return = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            x1 = int(detections[0, 0, i, 3] * self.frame_width)
            y1 = int(detections[0, 0, i, 4] * self.frame_height)
            x2 = int(detections[0, 0, i, 5] * self.frame_width)
            y2 = int(detections[0, 0, i, 6] * self.frame_height)
            if confidence > conf_threshold and x1 <= self.frame_width and x2 <= self.frame_width and y1 <= self.frame_height and y2 <= self.frame_height:
                face_boxes_to_return.append([x1, y1, x2, y2])
                cv2.rectangle(frame_with_drown_squares, (x1, y1), (x2, y2), (0, 0, 255), 2, 8)
        return frame_with_drown_squares, face_boxes_to_return

    def detect_single_frame(self, frame, a_name=None):
        result_img, face_boxes = self.__highlight_face(frame, a_name)
        height = frame.shape[0]
        width = frame.shape[1]
        if not face_boxes:
            return '0f-0m', frame
        f = 0
        m = 0
        for idx, face_box in enumerate(face_boxes):
            x_from = max(0, face_box[1] - self.padding)
            x_to = min(face_box[3] + self.padding, height - 1)
            y_from = max(0, face_box[0] - self.padding)
            y_to = min(face_box[2] + self.padding, width - 1)
            face = frame[x_from:x_to, y_from:y_to]
            gender_bool = self.get_gender_and_draw(face, result_img, face_box)
            if gender_bool:
                f = f + 1
            else:
                m = m + 1

        return str(f) + "f-" + str(m) + "m", result_img

    def detect_for_colab(self, frame, a_name=None):
        return self.detect_single_frame(frame, a_name)

    def detect_single_image(self, img_path):
        basename = os.path.basename(img_path)
        file_name, _ = os.path.splitext(basename)
        frame = cv2.imread(img_path)
        return self.detect_single_frame(frame, file_name)

    def detect_single_frame_counting(self, frame, a_name=None):
        result_img, face_boxes = self.__highlight_face(frame)
        if not face_boxes:
            return '0f-0m'
        height = frame.shape[0]
        width = frame.shape[1]
        f = 0
        m = 0
        for face_box in face_boxes:
            x_from = max(0, face_box[1] - self.padding)
            x_to = min(face_box[3] + self.padding, height - 1)
            y_from = max(0, face_box[0] - self.padding)
            y_to = min(face_box[2] + self.padding, width - 1)
            face = frame[x_from:x_to, y_from:y_to]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.gender_model_mean_values, swapRB=False)
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            if gender == 'f':
                f = f + 1
            elif gender == 'm':
                m = m + 1

            if a_name is not None:
                cv2.putText(result_img, f'{gender.upper()}', (face_box[0], face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(f'/Users/aneira/noticias/data/{a_name}_processed.png', result_img,
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return str(f) + "f-" + str(m) + "m"

    def set_hyperparameter(self, key, value):
        self.hyperparameters[key] = value

    def get_hyperparameters(self):
        return self.hyperparameters

    def set_padding(self, value):
        self.padding = value

    def get_padding(self):
        return self.padding
