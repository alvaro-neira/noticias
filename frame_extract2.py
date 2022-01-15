# Took 3:40 minutes, 1 frame per second, 1-hour video, without displaying the video. 3540 frames.

from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2


def process_frame(numpy_frame, counter):
    if counter == 15600 or counter == 15000 or counter == 15900:
        # cv2.imshow("video", numpy_frame)
        cv2.imwrite("data/frame_{}.png".format(counter), numpy_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream('data/tv24horas_2021_11_26_22.mp4').start()
time.sleep(1.0)

# start the FPS timer
fps = FPS().start()

count = 0
# loop over frames from the video file stream
while fvs.more():
    frame = fvs.read()
    if frame is None:
        break
    process_frame(frame, count)
    if count == 15900:
        break
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
