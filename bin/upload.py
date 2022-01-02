#!/Library/Frameworks/Python.framework/Versions/3.9/bin/python3

from google_drive.gdrive_folder import GDriveFolder

# gd = GDriveFolder("1AbWZDZI-4VHyKaCtoJolzy5QxQgylCRq")
gd = GDriveFolder("1cdm7j1VkdIim6GtjZIon34vMq5ahrg3v")

gd.oauth2_connect()
gd.upload('/Users/aneira/noticias/small.mp4')

print("done")
