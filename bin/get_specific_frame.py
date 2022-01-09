#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import sys
import os
from imutils.video import FileVideoStream
import time
import cv2

if len(sys.argv) != 3:
    print(f"Usage: % get_specific_frame.py <video filename> <frame number>")
    exit(-1)

basename = os.path.basename(sys.argv[1])
file_name, _ = os.path.splitext(basename)

fvs = FileVideoStream(sys.argv[1]).start()
time.sleep(1.0)

count = 0
# loop over frames from the video file stream
while fvs.more():
    frame = fvs.read()
    if frame is None:
        break
    if count == int(sys.argv[2]):
        cv2.imwrite(f"{file_name}_frame_{count}.png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        break
    # There is no waitKey()
    count = count + 1

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()

print("done")
