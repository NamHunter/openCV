import cv2
from matplotlib import pyplot as plt

img = cv2.imread('image\lena.jpg', 1)

cv2.imshow('anh', img)

plt.imshow(img)
plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
