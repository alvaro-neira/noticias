#!/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
import sys

from google_drive.gdrive_folder import GDriveFolder

if len(sys.argv) != 2:
    print(f"Usage: % upload.py <full filename>")
    exit(-1)
gd = GDriveFolder("1AbWZDZI-4VHyKaCtoJolzy5QxQgylCRq") # 24 folder
# gd = GDriveFolder("1cdm7j1VkdIim6GtjZIon34vMq5ahrg3v")  # test_upload

gd.oauth2_connect()
gd.upload(sys.argv[1])

print("done")
