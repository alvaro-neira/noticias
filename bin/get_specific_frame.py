#!/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
import sys
from imutils.video import FileVideoStream
import time
import cv2

if len(sys.argv) != 3:
    print(f"Usage: % get_specific_frame.py <video filename> <frame number>")
    exit(-1)

fvs = FileVideoStream(sys.argv[1]).start()
time.sleep(1.0)

count = 0
# loop over frames from the video file stream
while fvs.more():
    frame = fvs.read()
    if frame is None:
        break
    if count == int(sys.argv[2]):
        cv2.imwrite("frame_{}.png".format(count), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        break
    # There is no waitKey()
    count = count + 1

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()

print("done")
