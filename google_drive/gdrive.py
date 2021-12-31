from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth('settings.yml')
gauth.LocalWebserverAuth()  # client_secrets.json need to be in the same directory as the script
gauth.SaveCredentialsFile("google-drive-with-python.json")
drive = GoogleDrive(gauth)

fileID = '1cdm7j1VkdIim6GtjZIon34vMq5ahrg3v'
"""
This ID is for MY folder test_upload
"""
fileList = drive.ListFile({'q': "'" + fileID + "' in parents and trashed=false"}).GetList()
for file in fileList:
    print('Title: %s, ID: %s' % (file['title'], file['id']))

# Initialize GoogleDriveFile instance with file id.
file1 = drive.CreateFile({"mimeType": "text/csv", "parents": [{"kind": "drive#fileLink", "id": fileID}]})
file1.SetContentFile("small_file.csv")
file1.Upload()  # Upload the file.
print('Created file %s with mimeType %s' % (file1['title'], file1['mimeType']))
