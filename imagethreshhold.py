import cv2
import numpy as numpy
from matplotlib import pyplot as plt
img = cv2.imread('image\\gradient.png', 0)

a, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
a, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# sau giá trị 0 sẽ giữ lại 0
a, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)

titles = ['binary', 'b_inverse', 'trunc']
images = [th1, th2, th3]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], cmap='spring')
    plt.title(titles[0])
    plt.xticks([]), plt.yticks([])
plt.show()


# th4 = cv2.adaptiveThreshold(img, 255, )
# cv2.imshow('anh2', th3)
# cv2.imshow('anh', img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
