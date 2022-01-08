import os
import cv2
import csv
import random
import numpy as np
from imutils.video import FileVideoStream
import time
from google_drive.gdrive_folder import GDriveFolder

n_samples = 20
gd = GDriveFolder(GDriveFolder.folder_24)
gd.oauth2_connect()
file_list = gd.ls(do_print=False)

data_folder = '/Users/aneira/noticias/data/'
with open(data_folder + 'images.csv', 'w', encoding='UTF8', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for i in range(n_samples):
        the_file = random.choice(file_list)
        file_name = the_file['title']
        gd.oauth2_connect()
        gd.download_by_id(the_file['id'], data_folder)

        rand = np.random.randint(0, 3540 * 60)
        base_name, _ = os.path.splitext(file_name)
        fvs = FileVideoStream(data_folder + file_name).start()
        time.sleep(1.0)
        count = 0
        # loop over frames from the video file stream
        while fvs.more():
            frame = fvs.read()
            if frame is None:
                break
            if count == rand:
                cv2.imwrite(f"{data_folder}{base_name}_frame_{count}.png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                break
            count = count + 1

        # do a bit of cleanup
        cv2.destroyAllWindows()
        fvs.stop()

        gd2 = GDriveFolder(GDriveFolder.folder_img)

        gd2.oauth2_connect()
        id_str = gd2.upload(f"{data_folder}{base_name}_frame_{count}.png")
        os.remove(f"{data_folder}{base_name}_frame_{count}.png")
        os.remove(f"{data_folder}{file_name}")
        link_str = f"https://drive.google.com/file/d/{id_str}/view?usp=sharing"
        writer.writerow([file_name, count, 0, 0, '', 1, '', link_str])
print("done")
