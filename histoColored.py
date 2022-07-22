"""
    We can plot the histogram for colored channels as well
"""

from cProfile import label
import cv2
import matplotlib.pyplot as plt
import numpy as np

im= cv2.imread('img.png',cv2.IMREAD_COLOR)
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

clr_channels = {'red','green','blue'}

intensity_values = np.array([x for x in range(256)])

plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.title('Curve Plot')
for i,color in enumerate(clr_channels):
    hist = cv2.calcHist([im],[i],None,[256],[0,256])
    plt.plot(intensity_values,hist,color=color,label=color+' channel')

plt.subplot(1,2,2)
plt.title('Bar histogram')
for i,col in enumerate(clr_channels):
    hist = cv2.calcHist([im],[i],None,[256],[0,256])
    plt.bar(intensity_values,hist[:,0],color=col,width=4)

plt.tight_layout()
plt.show()