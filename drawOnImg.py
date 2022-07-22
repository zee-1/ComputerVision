from turtle import color
from PIL import Image,ImageOps,ImageDraw
import numpy as np
import matplotlib.pyplot as plt

img  = Image.open('img.png')
img_ = np.array(img)
img_arr = np.array(img)
img_draw = img.copy()

upp = 100
bott = 400
left = 100
rght = 400

img_arr[upp:bott,left:rght,1:2] = 0

plt.subplot(1,3,1)
plt.imshow(img_arr)
plt.subplot(1,3,2)
plt.imshow(img_)

img_fn = ImageDraw.Draw(im=img_draw)

side = [upp,left,bott,rght]

img_fn.rectangle(xy=side,fill='red')
plt.subplot(1,3,3)
plt.imshow(img_draw)

plt.show()