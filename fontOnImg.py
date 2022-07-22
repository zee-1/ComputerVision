from PIL import Image,ImageFont,ImageDraw
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('img.png')
img_arr = np.array(img)

img_draw = img.copy()

img_fn = ImageDraw.Draw(im = img_draw)

img_fn.text(xy=(100,400),text='Hagamoro',fill=(255,255,255))

plt.imshow(img_draw)
plt.show()

