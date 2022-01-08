import os
import json
import csv
import re

data_path = '/Users/aneira/noticias/data/'
header = ['Fecha', 'Vacio', 'Femenino', 'Masculino']


def get_stats(numbers_dict):
    total = 0
    vacio = 0
    femenino = 0
    masculino = 0
    for numbers in numbers_dict:
        if numbers == 'frames_total':
            continue
        print(numbers)
        n_frames = numbers_dict[numbers]
        total = total + n_frames
        res = re.findall('([0-9]+)f-([0-9]+)m', numbers)
        f = int(res[0][0])
        m = int(res[0][1])
        if f == 0 and m == 0:
            vacio = n_frames
        femenino = f * n_frames + femenino
        masculino = m * n_frames + masculino

    return vacio * 100.0 / total, femenino * 100.0 / total, masculino * 100.0 / total


json_file = open(data_path + 'news_stats.json')
data = json.load(json_file)
with open(data_path + 'news_stats.csv', 'w', encoding='UTF8', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)
    for video_file_name in data:
        file_name, _ = os.path.splitext(video_file_name)
        date_tokens = file_name.split('_')
        size = len(date_tokens)
        no_faces, fem, masc = get_stats(data[video_file_name])
        writer.writerow(
            [date_tokens[size - 4] + "-" + date_tokens[size - 3] + "-" + date_tokens[size - 2], no_faces, fem, masc])
# Closing file
json_file.close()
print("done")
