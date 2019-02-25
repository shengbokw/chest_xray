import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


path = "~/venv/IM-0001-0001.jpeg"
# newImage = open_image(path)
# greyscale = convert_grayscale(newImage)

img = cv2.imread('IM-0001-0001.jpeg',0)

edges = cv2.Canny(img,80,80)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()