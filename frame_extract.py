import cv2

capture = cv2.VideoCapture("data/small.mp4")
if not capture.isOpened():
    print("error!")
cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
cont = 0
while capture.grab():
    retval, frame_color = capture.retrieve()
    cv2.imshow("video", frame_color)
    if cont % 60 == 0:
        cv2.imwrite("frame_{}.png".format(cont), frame_color, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    key = cv2.waitKey(10)
    if key == ord('q') or key == 27:
        break
    cont += 1
capture.release()
cv2.destroyAllWindows()
