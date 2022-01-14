from imutils.video import FileVideoStream
from imutils.video import FPS
import time
import cv2
import pandas as pd
import random
import shutil

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

data_path = '/Users/aneira/noticias/data/'

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


def test_model_one(model, the_criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = the_criterion(outputs, labels)

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


def process_frame(numpy_frame, counter):
    print(f"counter={counter}")
    file_name = "{:05d}".format(counter)
    cv2.imwrite(f'{data_path}/frames/orig_{file_name}.png', numpy_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    result_local = evaluate_imagen(f'{data_path}/frames/orig_{file_name}.png', model_ft, criterion)
    if result_local == 'Noticias':
        result_img2 = cv2.cvtColor(cv2.imread(f'{data_path}/frames/orig_{file_name}.png'), cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'{data_path}/frames/bw_{file_name}.png', result_img2, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        result_img = cv2.imread(f'{data_path}/frames/bw_{file_name}.png')
    else:
        result_img = numpy_frame.copy()

    height = result_img.shape[0]
    width = result_img.shape[1]
    cv2.putText(result_img, result_local, (100, round(height / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 2,
                (255, 255, 255), 2, cv2.LINE_AA)
    final_frame = cv2.hconcat((numpy_frame, result_img))
    cv2.imwrite(f'{data_path}/frames/{file_name}.png', final_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(result_local)
    print('\n\n')


# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(data_path + "tv24horas_2021_11_26_22.mp4").start()
time.sleep(1.0)

# start the FPS timer
fps = FPS().start()

count = 0
# loop over frames from the video file stream
while fvs.more():
    frame = fvs.read()
    if frame is None:
        break
    if count > 9000:
        break
    process_frame(frame, count)
    # There is no waitKey()
    fps.update()
    count = count + 1

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()

print('done')
