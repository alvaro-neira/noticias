import pandas as pd
import random
import shutil

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import csv

random.seed(30)
device = 'cpu'
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

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)


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

    preds_cpu = preds.cpu()
    preds_cpu_numpy = preds_cpu.numpy()
    if preds_cpu_numpy[0] == 1:
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

df_test = pd.read_csv('/Users/aneira/noticias/ad_detection/resultados.csv')
with open('/Users/aneira/noticias/ad_detection/resultados_local.csv', 'w', encoding='UTF8', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['id', 'Nombre Final', 'Clase', 'result_red', 'correct', 'result_local'])
    for index, row in df_test.iterrows():
        print(row['Nombre Final'])
        result_local = evaluate_imagen(mypath + row['Nombre Final'] + '.png', model_ft, criterion)
        writer.writerow([index, row['Nombre Final'], row['Clase'], row['result_red'], row['correct'], result_local])

print('done')
