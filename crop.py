from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('img.png')
img_arr = np.array(img)

upper  = 100
bottom = 500
left = 150
right = 450

plt.imshow(img_arr[upper:bottom,left:right,:])
plt.show()

# use default crop func
plt.imshow(img.crop((left,upper,right,bottom)))
plt.show()