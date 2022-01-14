import os
import cv2

from hyper_face_classifier import HyperFaceClassifier

img_path = '/Users/aneira/noticias/data/2021_11_01_13_F6420.png'
hfc = HyperFaceClassifier('/Users/aneira/noticias/Gender-and-Age-Detection/opencv_face_detector_uint8.pb',
                          '/Users/aneira/noticias/Gender-and-Age-Detection/opencv_face_detector.pbtxt',
                          '/Users/aneira/hyperface/model_epoch_190')
basename = os.path.basename(img_path)
file_name, _ = os.path.splitext(basename)
frame = cv2.imread(img_path)
ret_str, result_frame = hfc.detect_single_frame(frame, file_name)
print(ret_str)
cv2.imwrite(f'/Users/aneira/noticias/data/{file_name}_processed.png', result_frame,
            [cv2.IMWRITE_PNG_COMPRESSION, 0])
