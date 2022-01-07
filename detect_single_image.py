import cv2
import os

hyperparameters = {'conf_threshold': 0.7}
# img_path = '/Users/aneira/noticias/data/tv24horas_2022_01_06_10_frame_124425.png'
img_path = '/Users/aneira/noticias/data/tv24horas_2022_01_06_12_frame_27000.png'
# img_path = '/Users/aneira/noticias/data/tv24horas_2022_01_06_15_frame_20423.png'

faceProto = "Gender-and-Age-Detection/opencv_face_detector.pbtxt"
faceModel = "Gender-and-Age-Detection/opencv_face_detector_uint8.pb"
genderProto = "Gender-and-Age-Detection/gender_deploy.prototxt"
genderModel = "Gender-and-Age-Detection/gender_net.caffemodel"

basename = os.path.basename(img_path)
file_name, _ = os.path.splitext(basename)


def save_blob(blob_local, full_path_png):
    blob_to_save = blob_local.reshape(blob_local.shape[2] * blob_local.shape[1], blob_local.shape[3], 1)
    print(blob_to_save.shape)
    cv2.imwrite(full_path_png, blob_to_save, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def highlight_face(face_net, the_frame, conf_threshold=hyperparameters['conf_threshold']):
    """
    This function is used only to detect faces (not gender)
    """
    frame_with_drown_squares = the_frame.copy()
    frame_height = frame_with_drown_squares.shape[0]
    frame_width = frame_with_drown_squares.shape[1]
    # They say that a mean = [104, 117, 123] is a standard and doesn't need to be changed nor calculated
    blob_local = cv2.dnn.blobFromImage(frame_with_drown_squares, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob_local)
    detections = face_net.forward()
    face_boxes_to_return = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes_to_return.append([x1, y1, x2, y2])
            cv2.rectangle(frame_with_drown_squares, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
    with open("/Users/aneira/noticias/Gender-and-Age-Detection/face_net.dumped", "w") as text_file:
        text_file.write(face_net.dump())
    return frame_with_drown_squares, face_boxes_to_return


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # Where are these from?
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

padding = 20
frame = cv2.imread(img_path)

resultImg, face_boxes = highlight_face(faceNet, frame)
if not face_boxes:
    print("No face detected")

for faceBox in face_boxes:
    face = frame[max(0, faceBox[1] - padding):
                 min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                                                                :min(faceBox[2] + padding, frame.shape[1] - 1)]

    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    gender_preds = genderNet.forward()
    gender = genderList[gender_preds[0].argmax()]
    print(f'Gender: {gender}')

    cv2.putText(resultImg, f'{gender}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(f'/Users/aneira/noticias/data/{file_name}_processed.png', resultImg, [cv2.IMWRITE_PNG_COMPRESSION, 0])
