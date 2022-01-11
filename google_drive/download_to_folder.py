from google_drive.gdrive_folder import GDriveFolder

gd = GDriveFolder(GDriveFolder.folder_24)

gd.oauth2_connect()
gd.download_by_name('tv24horas_2021_12_13_21.mp4', '/Users/aneira/noticias/data')

print("done")
