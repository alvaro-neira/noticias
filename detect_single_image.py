from gender_and_age import GenderAndAge
import os
import cv2

img_path = '/Users/aneira/noticias/data/tv24horas_2021_10_30_18_frame_128163.png'

gaa = GenderAndAge('/Users/aneira/noticias/Gender-and-Age-Detection')
basename = os.path.basename(img_path)
file_name, _ = os.path.splitext(basename)
frame = cv2.imread(img_path)
gaa.detect_for_colab(frame, file_name)
