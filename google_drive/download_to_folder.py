from google_drive.gdrive_folder import GDriveFolder

gd = GDriveFolder(GDriveFolder.folder_24)

gd.oauth2_connect()
gd.download_by_name('tv24horas_2022_01_08_09.mp4', '/Users/aneira/noticias/data')

print("done")
