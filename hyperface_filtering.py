import cv2
import os
import numpy as np
from dnn_model import DnnModel
import chainer


class HyperfaceFiltering(DnnModel):
    def __init__(self, opencv_pbtxt_path, opencv_pb_path, hyperface_model_path):
        os.environ["CHAINER_TYPE_CHECK"] = "0"
        self.face_proto = opencv_pbtxt_path
        self.face_model = opencv_pb_path
        self.face_net = cv2.dnn.readNetFromTensorflow(self.face_model, self.face_proto)
        self.padding = 20
        self.gender_list = ['M', 'F']
        self.hyperface_threshold = 0.004
        self.hyperface_model = models.HyperFaceModel()
        self.hyperface_model.train = False
        self.hyperface_model.report = False
        self.hyperface_model.backward = False

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

    def detect_single_image(self, img_path):
        pass

    def detect_single_frame(self, frame, a_name):



        # Initialize model
        chainer.serializers.load_npz(model_path, model)

        # Setup GPU
        if config.gpu >= 0:
            chainer.cuda.check_cuda_available()
            chainer.cuda.get_device(config.gpu).use()
            model.to_gpu()
            xp = chainer.cuda.cupy
        else:
            xp = np

        # Load image file
        img = cv2.imread(img_path)
        if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            exit()
        img = img.astype(np.float32) / 255.0  # [0:1]
        img = cv2.resize(img, models.IMG_SIZE)
        img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX)
        img = np.transpose(img, (2, 0, 1))

        # Create single batch
        imgs = xp.asarray([img])
        x = chainer.Variable(imgs)  # , volatile=True)

        # Forward
        y = model(x)

        # Chainer.Variable -> np.ndarray
        detection = _cvt_variable(y['detection'])
        genders = _cvt_variable(y['gender'])

        gender = genders[0]

        if gender > 0.5:
            return f"Female, detection={detection}"
        else:
            return f"Male, detection={detection}"

    def detect_for_colab(self, frame, a_name):
        pass
