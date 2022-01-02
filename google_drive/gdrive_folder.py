import json
import os
from json import JSONDecodeError

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


class GDriveFolder:
    """A comment"""
    folder_id = None
    google_drive = None
    base_path = '/Users/aneira/noticias'

    def __init__(self, folder_id):
        self.folder_id = folder_id

    def oauth2_connect(self):
        gauth = GoogleAuth()
        gauth.settings['client_config_file'] = self.base_path + "/google_drive/client_secrets.json"
        gauth.settings['save_credentials_file'] = self.base_path + "/google_drive/gdrive_folder.json"
        data = {}
        f = None
        try:
            f = open(self.base_path + "/google_drive/gdrive_folder.json")
            data = json.load(f)
        except JSONDecodeError as e:
            print("gdrive_folder.json empty")

        if gauth.credentials is None:
            gauth.LoadCredentials('file')
        if 'refresh_token' in data and data['refresh_token'] is not None and len(data['refresh_token']) > 0:
            gauth.credentials.refresh_token = data['refresh_token']
            gauth.GetFlow()
        gauth.LocalWebserverAuth()  # client_secrets.json need to be in the same directory as the script
        gauth.SaveCredentialsFile(self.base_path + "/google_drive/gdrive_folder.json")
        self.google_drive = GoogleDrive(gauth)
        if f is not None:
            f.close()
        print("oauth2 done")

    def ls(self):
        file_list = self.google_drive.ListFile({'q': "'" + self.folder_id + "' in parents and trashed=false"}).GetList()
        file_list.sort(key=lambda elem: elem['title'], reverse=True)
        for file in file_list:
            print(f"{file['title']},{file['fileSize']}")

    def upload(self, file_path):
        file1 = self.google_drive.CreateFile(
            {"mimeType": self.get_mime_type(file_path), "parents": [{"kind": "drive#fileLink", "id": self.folder_id}]})
        file1['title'] = os.path.basename(file_path)
        file1.SetContentFile(file_path)
        file1.Upload()  # Upload the file.
        print('Created file %s with mimeType %s' % (file1['title'], file1['mimeType']))

    @staticmethod
    def get_mime_type(file_name):
        _, file_extension = os.path.splitext(file_name)
        if len(file_extension) < 1:
            return None
        fext = file_extension.lower()
        if fext == 'csv':
            return 'text/csv'
        elif fext == 'mp4':
            return 'video/mp4'
        return None

    @staticmethod
    def get_by_title(file_list, file_title):
        for file in file_list:
            if file['title'] == file_title:
                return file['id']
        return None

    def download_by_name(self, name_string):
        file_list = self.google_drive.ListFile({'q': "'" + self.folder_id + "' in parents and trashed=false"}).GetList()
        file_id = self.get_by_title(file_list, name_string)
        if file_id is None:
            print(f"'{name_string}' not found")
            return
        file_to_download = self.google_drive.CreateFile({'id': file_id})
        file_to_download.GetContentFile(file_to_download['title'])

    def display_gdrive_folder(self):
        print("Folder ID: ", self.folder_id)
