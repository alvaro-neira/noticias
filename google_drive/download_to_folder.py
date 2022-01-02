from google_drive.gdrive_folder import GDriveFolder

gd = GDriveFolder("1AbWZDZI-4VHyKaCtoJolzy5QxQgylCRq")

gd.oauth2_connect()
gd.download_by_name('tv24horas_2022_01_02_14.mp4', '/Users/aneira/noticias/data')

print("done")
