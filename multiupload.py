import sys
import os
from google_drive.gdrive_folder import GDriveFolder
gd = GDriveFolder('17nRobcEcyCAw5fbyh4H3dtQr_E7fml1o')
gd.oauth2_connect()

for filename in os.listdir("/Users/aneira/noticias/data"):
    if not filename.endswith(".png"):
        continue
    img_path = os.path.join("/Users/aneira/noticias/data", filename)
    gd.upload(img_path)
    print(img_path)


print("done")
