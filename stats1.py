import os
from google_drive.gdrive_folder import GDriveFolder
from gender_and_age import GenderAndAge
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2

data_path = '/Users/aneira/noticias/data/'
gaa = GenderAndAge('/Users/aneira/noticias/Gender-and-Age-Detection')


def process_video(video_path):
    fvs = FileVideoStream(video_path).start()
    time.sleep(1.0)
    count = 0
    while fvs.more():
        frame = fvs.read()
        if frame is None:
            break
        if count % 10800 == 0:
            print(f"count={count}")
            cv2.imwrite(f"{data_path}frame_{count}.png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            gaa.detect_single_frame(frame, f"frame_{count}")
        count = count + 1
    print(f"total count={count}")
    cv2.destroyAllWindows()
    fvs.stop()


gd = GDriveFolder("1AbWZDZI-4VHyKaCtoJolzy5QxQgylCRq")

gd.oauth2_connect()
file_list = gd.ls(do_print=False)
for file in file_list:
    file_name, _ = os.path.splitext(file['title'])
    tokens = file_name.split('_')
    if tokens[len(tokens) - 1] == '22':
        print(f'{file_name}')
        gd.download_by_id(file['id'], data_path)
        process_video(data_path + file['title'])
        break
print("done")
