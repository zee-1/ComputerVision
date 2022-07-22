import cv2
from cv2 import IMREAD_COLOR
from cv2 import IMREAD_GRAYSCALE
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread('img_gray.png',IMREAD_GRAYSCALE)


'''
    transformation type
        s => T{f[i,j]} => hs
    linear transformation g[i,j] = alpha*f[i,j] + beta
    if alpha = -1       and         beta = 255
        the image will become negative
    alpha: corresponds to the contrast
    beta: corresponds to brightness
'''

im_contra = 4*im

im_bright = im+50 

im_neg = -1*im + 255

plt.figure(figsize=(50,50))

plt.subplot(4,1,1)
plt.title('Contrast++')
plt.imshow(im_contra,cmap='gray')

plt.subplot(4,1,2)
plt.title('Negative')
plt.imshow(im_neg,cmap='gray')

plt.subplot(4,1,3)
plt.title('Original')
plt.imshow(im,cmap='gray')

plt.subplot(4,1,4)
plt.title('Bright')
plt.imshow(im,cmap='gray')

plt.show()