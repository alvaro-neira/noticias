import os
from google_drive.gdrive_folder import GDriveFolder
from gender_and_age import GenderAndAge
from imutils.video import FileVideoStream
import numpy as np
import imutils
import time
import cv2
import json

data_path = '/Users/aneira/noticias/data/'
gaa = GenderAndAge('/Users/aneira/noticias/Gender-and-Age-Detection')

json_data = {}


def process_video(full_folder, video_name):
    fvs = FileVideoStream(full_folder + video_name).start()
    time.sleep(1.0)
    count = 0
    qty_dict = {}
    while fvs.more():
        frame = fvs.read()
        if frame is None:
            break
        if count % 60 == 0:
            print(f"count={count}")
            key_str = gaa.detect_single_frame(frame)
            if key_str in qty_dict:
                qty_dict[key_str] = qty_dict[key_str] + 1
            else:
                qty_dict[key_str] = 1
        count = count + 1
    print(f"total count={count}")
    json_data[video_name] = qty_dict
    json_data[video_name]['frames_total'] = count
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
        process_video(data_path, file['title'])
        os.remove(data_path + file['title'])
        break

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)
print("done")
