from PIL import Image,ImageOps
import numpy as np
import matplotlib.pyplot as plt
img  = Image.open('img.png')
img_arr = np.array(img)

def upside_down(img_arr):
    upSideDown = np.zeros(img_arr.shape, dtype=np.uint8)
    for i,row in enumerate(img_arr):
        upSideDown[img_arr.shape[0]-i-1,:,:] = row
    return upSideDown

plt.figure(figsize=(10,10))
plt.imshow(upside_down(img_arr))
plt.show()

img_flipped = ImageOps.flip(img)
plt.imshow(img_flipped)
plt.show()

img_mirr = ImageOps.mirror(img)
plt.imshow(img_mirr)
plt.show()