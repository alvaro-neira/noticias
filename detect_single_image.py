from gender_and_age import GenderAndAge
import os
import cv2

img_path = '/Users/aneira/noticias/data/tv24horas_2021_11_26_22_5100.png'
gaa = GenderAndAge('/Users/aneira/noticias/Gender-and-Age-Detection')
basename = os.path.basename(img_path)
file_name, _ = os.path.splitext(basename)
frame = cv2.imread(img_path)
n, result_frame = gaa.detect_single_frame(frame, file_name)
print(n)
cv2.imwrite(f'/Users/aneira/noticias/data/{file_name}_processed.png', result_frame,
            [cv2.IMWRITE_PNG_COMPRESSION, 0])


