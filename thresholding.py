"""
    Thresholding is the concept of the painting pixels that are above threshold value as max_value else with min_value or they stay same 

    This is an important concept of the image segmentation
"""
import cv2
from cv2 import imread
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread('img_gray.png',cv2.IMREAD_GRAYSCALE)

def thresholding(img,threshold_value,min_val=0,max_val=255):
    n,m = img.shape
    new_img = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            if img[i,j]> threshold_value:
                new_img[i,j] = max_val
            else:
                new_img[i,j] = min_val

    return new_img

threshed = thresholding(im,threshold_value=95,min_val=0,max_val=255)

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(im,cmap='gray')

plt.subplot(1,2,2)
plt.title("Pixels with intensity greater than 95 are \n marked max_value:255 and rest min_value: 0")
plt.imshow(threshed,cmap='gray')
plt.tight_layout()
plt.show()
##      One can use the thresholding function provided by cv2
#   cv2.threshold(  img,  min_value,    max_value,  thresholding_type)
"""   
    thresholding type THRESH_BINARY gives the same result as above
                      THRESH_TRUNC  will not change the values if they are below threshold
                      THRESH_OTUS   Determine the best threshold value by looking at the histogram
"""

method = {cv2.THRESH_BINARY, cv2.THRESH_TRUNC, cv2.THRESH_OTSU}
im2 = imread('cameraman.jpeg',cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10,5))

for i,meth in enumerate(method):
    plt.subplot(1,3,i+1)
    plt.title('Method:'+str(meth))
    thresh = cv2.threshold(src=im2,thresh=50,maxval= 255,type= meth)[1]
    plt.imshow(thresh,cmap='gray')
plt.tight_layout()
# plt.imshow(im2,cmap='gray')
plt.show()
