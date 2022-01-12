import os
import time
import cv2
import re
from imutils.video import FileVideoStream
import time
from gender_and_age import GenderAndAge
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# define the scope
from google_sheets.gsheets import GSheets

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
# add credentials to the account
creds = ServiceAccountCredentials.from_json_keyfile_name('/Users/aneira/noticias/google_sheets/client_secret.json',
                                                         scope)
# authorize the clientsheet
gc = gspread.authorize(creds)
worksheet = gc.open('Copy of Etiquetado-Paula')
sheet_instance = worksheet.worksheet("2021-11-26-22 (0.3)")

gaa = GenderAndAge('/Users/aneira/noticias/Gender-and-Age-Detection')

video_file = 'tv24horas_2021_11_26_22.mp4'
n_det = 9
data_folder = '/Users/aneira/noticias/data/'
base_name, _ = os.path.splitext(video_file)
fvs = FileVideoStream(data_folder + video_file).start()
time.sleep(1.0)
count = 0

sheet_row = 2

letter1 = GSheets.change_base_excel(3)
letter2 = GSheets.change_base_excel(3 + n_det + 1)
letter_total_M = GSheets.change_base_excel(3 + n_det)
while fvs.more():
    frame = fvs.read()
    if frame is None:
        break
    if count > 113400:
        break
    if count % 300 == 0:
        print(f"frame {count}, sleeping 1")
        time.sleep(2)
        cv2.imwrite(f'/Users/aneira/noticias/data/{base_name}_{count}.png', frame,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])
        sheet_row = sheet_row + 1

        cell_list = sheet_instance.range(f'{letter1}{sheet_row}:{letter2}{sheet_row}')

        # detecciones
        for i in range(n_det):
            cell_list[i].value = ""
        res_str, result_img = gaa.detect_single_frame(frame, f"{base_name}_{count}")
        if (res_str == '0f-0m' and sheet_instance.acell(letter1 + str(sheet_row - 1)).value.strip() == 'N'
                and sheet_instance.acell(letter2 + str(sheet_row - 1)).value.strip() == '0'
                and sheet_instance.acell(letter_total_M + str(sheet_row - 1)).value.strip() == '0'):
            cell_list[0].value = "N"
            print(f"frame {count} ready")

        res = re.findall('([0-9]+)f-([0-9]+)m', res_str)
        f = int(res[0][0])
        m = int(res[0][1])
        cell_list[n_det].value = m
        cell_list[n_det + 1].value = f

        sheet_instance.update_cells(cell_list)
        cv2.imwrite(f'/Users/aneira/noticias/data/{base_name}_{count}_processed.png', result_img,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])

        sheet_row = sheet_row + 1

    count = count + 1

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()
print("done")
