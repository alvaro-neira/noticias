import os
import cv2
import csv
import re
import numpy as np
from imutils.video import FileVideoStream
import time
from google_drive.gdrive_folder import GDriveFolder
from gender_and_age import GenderAndAge

gd = GDriveFolder(GDriveFolder.folder_24)
gd.oauth2_connect()
video_file = 'tv24horas_2021_11_26_22.mp4'
data_folder = '/Users/aneira/noticias/data/'
gd.download_by_name(video_file, data_folder)
base_name, _ = os.path.splitext(video_file)
fvs = FileVideoStream(data_folder + video_file).start()
time.sleep(1.0)
count = 0
with open(data_folder + 'paula.csv', 'w', encoding='UTF8', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['frame', 'tipo', 'detecciones', 'detecciones', 'detecciones', 'detecciones', 'detecciones',
                     'total M', 'total F', 'total N', 'VP M', 'FP M', 'FN M', 'VP F', 'FP F', 'FN F', 'ref'])
    while fvs.more():
        frame = fvs.read()
        if frame is None:
            break
        if count > 113400:
            break
        if count % 300 == 0:
            cv2.imwrite(f"{data_folder}{base_name}_frame_{count}.png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            gd2 = GDriveFolder(GDriveFolder.folder_img)
            gd2.oauth2_connect()
            id_str = gd2.upload(f"{data_folder}{base_name}_frame_{count}.png")
            link_str = f"https://drive.google.com/file/d/{id_str}/view?usp=sharing"
            writer.writerow(
                [f"{base_name}_frame_{count}.png", 'real', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                 link_str])
            gaa = GenderAndAge('/Users/aneira/noticias/Gender-and-Age-Detection')
            res_str, new_frame = gaa.detect_single_image(f"{data_folder}{base_name}_frame_{count}.png")
            res = re.findall('([0-9]+)f-([0-9]+)m', res_str)
            f = int(res[0][0])
            m = int(res[0][1])
            cv2.imwrite(f"{data_folder}{base_name}_frame_{count}_processed.png", new_frame,
                        [cv2.IMWRITE_PNG_COMPRESSION, 0])
            gd2.oauth2_connect()
            id_str2 = gd2.upload(f"{data_folder}{base_name}_frame_{count}_processed.png")
            link_str2 = f"https://drive.google.com/file/d/{id_str2}/view?usp=sharing"
            writer.writerow(
                [f"{base_name}_frame_{count}.png", 'detectado', '', '', '', '', '', m, f, '', '', '', '', '', '', '',
                 link_str2])
            os.remove(f"{data_folder}{base_name}_frame_{count}.png")
            os.remove(f"{data_folder}{base_name}_frame_{count}_processed.png")
        count = count + 1

    # do a bit of cleanup
    cv2.destroyAllWindows()
    fvs.stop()
    os.remove(f"{data_folder}{video_file}")
    print("done")
