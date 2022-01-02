from google_drive.gdrive_folder import GDriveFolder

gd = GDriveFolder("1AbWZDZI-4VHyKaCtoJolzy5QxQgylCRq")

gd.oauth2_connect()
gd.ls()

print("done")
