import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# define the scope
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

# add credentials to the account
creds = ServiceAccountCredentials.from_json_keyfile_name('/Users/aneira/noticias/google_sheets/client_secret.json',
                                                         scope)

# authorize the clientsheet
gc = gspread.authorize(creds)

worksheet = gc.open('Copy of Etiquetado-Paula')
sheet_instance = worksheet.worksheet("2021-12-13-21 (0.3)")
print(sheet_instance)

print(f"total columns={sheet_instance.col_count}")

# records_data = sheet_instance.get_all_records()
#
sheet_instance.update_cell(1, 1, "asd")
sheet_instance.update_cell(2, 2, "bla")

