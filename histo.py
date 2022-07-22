import cv2
from cv2 import IMREAD_COLOR
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread('img.png',IMREAD_COLOR)
im_gray = cv2.cvtColor(im.astype('uint8'),cv2.COLOR_BGR2GRAY)
# im_gray = cv2.cvtColor(im_gray,cv2.COLOR_RGB2GRAY)

hist = cv2.calcHist([im],[0],None,[256],[0,255])
intensity_vals = np.array([x for x in range(256)])

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.title('image')
plt.imshow(im_gray,cmap='gray')

plt.subplot(1,2,2)
plt.title('histogram')
plt.bar(intensity_vals,hist[:,0],width=2)
plt.show()
