import cv2

originalImage = cv2.imread('/Users/aneira/noticias/data/frames/orig_00000.png')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

# cv2.imshow('Gray image', grayImage)
cv2.imwrite('/Users/aneira/noticias/data/frames/swa.png', grayImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# cv2.waitKey(0)
cv2.destroyAllWindows()
