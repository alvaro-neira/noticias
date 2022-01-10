from gender_and_age import GenderAndAge
import os
import cv2


img_path = '/Users/aneira/noticias/data/tv24horas_2021_11_09_18_frame_87580.png'
gaa = GenderAndAge('/Users/aneira/noticias/Gender-and-Age-Detection')
basename = os.path.basename(img_path)
file_name, _ = os.path.splitext(basename)
frame = cv2.imread(img_path)
n, result_frame = gaa.get_all_bounding_boxes(frame, 0, 230400)
print(n)
cv2.imwrite(f'/Users/aneira/noticias/data/{file_name}_processed.png', result_frame,
            [cv2.IMWRITE_PNG_COMPRESSION, 0])

