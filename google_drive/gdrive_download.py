import json
from json import JSONDecodeError

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.settings['save_credentials_file'] = 'gdrive_folder.json'
data = {}
try:
    f = open('gdrive_folder.json')
    data = json.load(f)
except JSONDecodeError as e:
    print("gdrive_folder.json empty")

if gauth.credentials is None:
    gauth.LoadCredentials('file')
if 'refresh_token' in data and len(data['refresh_token']) > 0:
    # gauth.flow = data['refresh_token']
    gauth.credentials.refresh_token = data['refresh_token']
    gauth.GetFlow()
gauth.LocalWebserverAuth()  # client_secrets.json need to be in the same directory as the script
gauth.SaveCredentialsFile("gdrive_folder.json")
drive = GoogleDrive(gauth)

folderId = '1cdm7j1VkdIim6GtjZIon34vMq5ahrg3v'
fileList = drive.ListFile({'q': "'" + folderId + "' in parents and trashed=false"}).GetList()
for file in fileList:
    print('Title: %s, ID: %s' % (file['title'], file['id']))

# filetodownload = drive.CreateFile({'id': '1XJZe-BMfTmQSkEH7EyEXUIVcfzWGKcKQ'})
# filetodownload.GetContentFile(filetodownload['title'])
# f.close()
print("done")
