import os
import cv2
from hyper_face_filtering import HyperFaceFiltering

img_path = '/Users/aneira/noticias/data/tv24horas_2021_11_26_22_5100.png'
hff = HyperFaceFiltering('/Users/aneira/noticias/Gender-and-Age-Detection/opencv_face_detector.pbtxt',
                         '/Users/aneira/noticias/Gender-and-Age-Detection/opencv_face_detector_uint8.pb',
                         '/Users/aneira/Downloads/model_epoch_190')
basename = os.path.basename(img_path)
file_name, _ = os.path.splitext(basename)
frame = cv2.imread(img_path)
n, result_frame = hff.detect_single_frame(frame)
print(f"ret_str={n}")
cv2.imwrite(f'/Users/aneira/noticias/data/{file_name}_processed.png', result_frame,
            [cv2.IMWRITE_PNG_COMPRESSION, 0])


