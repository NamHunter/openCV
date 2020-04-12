import cv2
import numpy as np


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        blue = img[x, y, 0]
        green = img[x, y, 1]
        red = img[x, y, 2]
        img2 = np.zeros([200, 200, 3], np.uint8)
        img2[:, :, 0] = blue
        img2[:, :, 1] = green
        img2[:, :, 2] = red
        # img2[:] = [blue, green, red]
        cv2.imshow("anh3", img2)


img = cv2.imread('image\lena.jpg', 1)
cv2.imshow('anh2', img)
cv2.setMouseCallback("anh2", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
