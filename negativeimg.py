"""
    In this document we will discuss the Linear Tranformation of a matrix,
    here, matrix is the np array of an image
    g[i,j] = T{ f[i,j] }
    f[i,j] => T => g[i,j]

    In genral 
    new_matrix = alpha * matrix + beta

    alpha => controls the contrast of the image
    beta => controls the brightness of the image
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread('img_gray.png',cv2.IMREAD_GRAYSCALE)

# to get a negative of an image alpha = -1 and beta = 255
# general formula s = L - 1 - r                             s: the new value, L: the maximum intensity, r: previous value
im_neg =  -1 * im + 255
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(im,cmap='gray')

plt.subplot(1,2,2)
plt.title('Negative')
plt.imshow(im_neg,cmap='gray')
plt.show()
# For linear Transformation the cv2 provides a method cv2.convertScaleAbs(img,alpha=alpha,beta=beta)
img_trans_cv2 = cv2.convertScaleAbs(im,alpha=2,beta=20)

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(im,cmap='gray')

plt.subplot(1,2,2)
plt.title('cv2 transformed')
plt.imshow(img_trans_cv2,cmap='gray')


plt.show()