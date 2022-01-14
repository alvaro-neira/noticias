from imutils.video import FileVideoStream
from imutils.video import FPS
import time
import cv2

from hyper_face_classifier import HyperFaceClassifier

data_path = '/Users/aneira/noticias/data/'

hfc = HyperFaceClassifier('/Users/aneira/noticias/Gender-and-Age-Detection/opencv_face_detector_uint8.pb',
                          '/Users/aneira/noticias/Gender-and-Age-Detection/opencv_face_detector.pbtxt',
                          '/Users/aneira/hyperface/model_epoch_190',
                          360,
                          640,
                          17)


def process_frame(numpy_frame, counter):
    print(f"counter={counter}")
    _, result_img = hfc.detect_single_frame(numpy_frame)
    final_frame = cv2.hconcat((numpy_frame, result_img))
    file_name = "{:05d}".format(counter)
    cv2.imwrite(f'{data_path}/frames/{file_name}.png', final_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])


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
