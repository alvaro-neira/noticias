#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import sys

from google_drive.gdrive_folder import GDriveFolder

if len(sys.argv) != 2:
    print(f"Usage: % upload.py <full filename>")
    exit(-1)
gd = GDriveFolder(GDriveFolder.folder_24)


gd.oauth2_connect()
gd.upload(sys.argv[1])

print("done")
