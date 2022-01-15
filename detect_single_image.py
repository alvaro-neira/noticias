from for_reports import ForReports
import os
import cv2

img_path = '/Users/aneira/noticias/data/frame_15900.png'
gaa = ForReports('/Users/aneira/noticias/Gender-and-Age-Detection')
basename = os.path.basename(img_path)
file_name, _ = os.path.splitext(basename)
frame = cv2.imread(img_path)
result_frame = gaa.detect_single_frame_blank(frame, file_name)
cv2.imwrite(f'/Users/aneira/noticias/data/{file_name}_processed.png', result_frame,
            [cv2.IMWRITE_PNG_COMPRESSION, 0])
