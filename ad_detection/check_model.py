from os import listdir
from os.path import isfile, join

import pandas as pd
import math
import random
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import time
import copy

random.seed(30)

# """Se exportan los datos con que se entreno el modelo, dejar así porque estos datos tienen las clases."""
#
# shutil.copy("/content/drive/MyDrive/Taller Diplomado IA/Data Comerciales/processed/data.zip", "/content/data.zip")
# shutil.copy("/content/drive/MyDrive/Taller Diplomado IA/Data Comerciales/processed/Comerciales.csv", "/content/Comerciales.csv")
# !unzip data.zip
#
# """El dataset de los datos con que se entreno el modelo."""
#
# df = pd.read_csv('Comerciales.csv', header=0, sep=";")
#
# df
#
# classesFinal = df["Clase"].unique()
# #classesFinal = [cl.replace(' ', '_') for cl in classes]
# print(classesFinal)
#
# """Se necesita de una sola carpeta, aquella que es de test. Esta carpeta se dejará vacía a excepción de una imagen en cada carpeta de cada clase. (Si no se hace esto se generá un error después)"""
#
# try:
#   os.mkdir('dataset')
# except OSError:
#   print ("No se pudo crear folder dataset")
# else:
#   print ("Se creó folder dataset")
#
#
# try:
#   os.mkdir('dataset/test')
# except OSError:
#   print ("No se pudo crear folder dataset")
# else:
#   print ("Se creó folder dataset")
#
#
#
# for cl in classesFinal:
#
#
#   try:
#     os.mkdir(os.path.join('dataset', "test", cl))
#   except OSError:
#     print (f"No se pudo crear folder test {cl}")
#   else:
#     print (f"Se creó folder test {cl}")
#
# dataset = dict()
#
# for index, cl in enumerate(classesFinal):
#     grouped_data = df.groupby(by=["Clase"]).get_group(cl)["Nombre Final"].tolist()
#     dataset[classesFinal[index]] = grouped_data
#
# for k,v in dataset.items():
#     print(f'Class: {k}, Length: {len(v)}')
#
#     print(v)
#

#
# shutil.copy("/content/drive/MyDrive/Taller Diplomado IA/Data Comerciales/resnet18_finetuned.pth", "/content/resnet18_finetuned.pth")
#
path_dataset = '/Users/aneira/noticias/ad_detection/dataset/'
mypath = '/Users/aneira/noticias/ad_detection/data_test/'
path_data = '/Users/aneira/noticias/ad_detection/data/'

shutil.copy("/Users/aneira/noticias/ad_detection/Hola.jpg", path_dataset + "test/Noticias/Hola.jpg")
shutil.copy("/Users/aneira/noticias/ad_detection/Hola.jpg", path_dataset + "test/Comercial/Hola2.jpg")

test_dataset = torchvision.datasets.ImageFolder(path_dataset + 'test',
                                                transform=transforms.Compose([transforms.Resize(224),
                                                                              transforms.ToTensor(),
                                                                              transforms.Normalize(
                                                                                  mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225])]))

#
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

#
# class_names = test_dataset.classes
#
device = 'cpu'


def test_model_one(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_dataset)
    epoch_acc = running_corrects / len(test_dataset)
    # print('Test Loss: {:.4f}  Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    predscpu = preds.cpu()
    predscpuNumpy = predscpu.numpy()
    if predscpuNumpy[0] == 1:
        prediccion = "Noticias"
    else:
        prediccion = "Comercial"

    return prediccion


def evaluate_imagen(imagedir, model, the_criterion):
    shutil.copy2(imagedir, path_dataset + "test/Noticias/Hola.jpg")
    shutil.copy2(imagedir, path_dataset + "test/Comercial/Hola2.jpg")
    return test_model_one(model, the_criterion)


model_ft = models.resnet18(pretrained=True)
num_ft = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ft, 2)
model_ft.load_state_dict(torch.load("/Users/aneira/noticias/ad_detection/resnet18_finetuned.pth",
                                    map_location=torch.device('cpu')))
criterion = nn.CrossEntropyLoss()

# ret_class = evaluate_imagen(path_data + "2021_10_27_09_F5100.jpg", model_ft, criterion)
# print(ret_class)

#
# evaluate_imagen("/content/drive/MyDrive/Taller Diplomado IA/Resultados/resultados_09/result_2021_11_26_22_frame_102600.png",model_ft,criterion)
#
# shutil.copy("/content/drive/MyDrive/Taller Diplomado IA/Data Comerciales/processed/Comerciales_test.csv", "/content/Comerciales_test_v2.csv")
#
# shutil.copy("/content/drive/MyDrive/Taller Diplomado IA/Data Comerciales/processed/data_test.zip", "/content/data_test.zip")
# shutil.copy("/content/drive/MyDrive/Taller Diplomado IA/Data Comerciales/processed/Comerciales_test.csv", "/content/Comerciales_test.csv")
# !unzip data_test.zip
#
# from os import listdir
# from os.path import isfile, join

ls_img = [f for f in listdir(mypath) if isfile(join(mypath, f))]

compare = []
img_list = []
for image in ls_img:
    result = evaluate_imagen(mypath + image, model_ft, criterion)
    print(f"{result} {mypath}{image}")
    compare.append(result)
    img_list.append(image[:-4])
#
# compare
#
# df_test = pd.read_csv('Comerciales_test.csv', header=0, sep=",")
# df_red = pd.DataFrame()
# df_red['result_red'] = compare
# df_red['Nombre Final'] = img_list
# df_test = pd.merge(df_test, df_red, how='left', on='Nombre Final')
# df_test.head()
#
# print(len(df_test))
#
# conditions = [
#     (df_test['Clase'] == 'Noticias') & (df_test['result_red'] == 'Noticias'),
#     (df_test['Clase'] == 'Comercial') & (df_test['result_red'] == 'Comercial'),
#     (df_test['Clase'] == 'Noticias') & (df_test['result_red'] == 'Comercial'),
#     (df_test['Clase'] == 'Comercial') & (df_test['result_red'] == 'Noticias'),
#     ]
# choices = [1, 1, 0, 0]
# df_test['correct'] = np.select(conditions, choices,)
# print(df_test['correct'].sum())
#
# df_test.to_csv('resultados.csv')
#
