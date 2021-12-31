import json
from json import JSONDecodeError

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

folder24 = '1AbWZDZI-4VHyKaCtoJolzy5QxQgylCRq'


def get_last_file(file_list):
    if len(file_list) < 1:
        return None
    ret_val = file_list[0]
    for file in file_list:
        print('Title: %s, ID: %s' % (file['title'], file['id']))
        if ret_val['title'] < file['title']:
            ret_val = file
    return ret_val['id']


def get_by_title(file_list, file_title):
    for file in file_list:
        print('Title: %s, ID: %s' % (file['title'], file['id']))
        if file['title'] == file_title:
            return file['id']
    return None


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
    gauth.credentials.refresh_token = data['refresh_token']
    gauth.GetFlow()
gauth.LocalWebserverAuth()  # client_secrets.json need to be in the same directory as the script
gauth.SaveCredentialsFile("gdrive_folder.json")
drive = GoogleDrive(gauth)

fileList = drive.ListFile({'q': "'" + folder24 + "' in parents and trashed=false"}).GetList()
file_id = get_by_title(fileList, 'tv24horas_2021_12_19_11.mp4')
file_to_download = drive.CreateFile({'id': file_id})
file_to_download.GetContentFile(file_to_download['title'])
f.close()
print("done")



