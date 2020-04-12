import cv2
from matplotlib import pyplot as plt

# img = cv2.imread('image\smarties.png', 0)
# # cv2.imshow('a', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# _, mark = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

# titles = ['image', 'mark']
# images = [img, mark]

# for i in range(2):
#     plt.subplot(2, 1, i+1)
#     plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

print(cv2.__file__)
