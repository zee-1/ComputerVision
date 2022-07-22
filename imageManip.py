from PIL import Image,ImageOps
import numpy as np
import matplotlib.pyplot as plt


img  = Image.open('img.png')
img_arr = np.array(img)
print(img_arr.shape)
print(np.transpose(img_arr,axes=(1,0,2)).shape)
img_trans = np.transpose(img_arr,axes=(1,0,2))
plt.imshow(img_trans)
plt.show()