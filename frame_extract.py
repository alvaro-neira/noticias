import cv2

capture = cv2.VideoCapture("data/small.mp4")
if not capture.isOpened():
    print("error!")
cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)


def process_frame(numpy_frame, counter):
    cv2.imshow("video", numpy_frame)
    if counter % 60 == 0:
        cv2.imwrite("data/frame_{}.png".format(counter), numpy_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])


count = 0
while capture.grab():
    _, frame_color = capture.retrieve()
    process_frame(frame_color, count)
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break
    count += 1
capture.release()
cv2.destroyAllWindows()
