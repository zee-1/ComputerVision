"""
    Geometric Operation includes:
    Scaling
    Translating
    Rotating

    Scaling: Extending an image in either 1 direction or both direction
        say to increase the image size to 2x in X direction
        g[i,2j] = f[i,j]
            this means that the jth row of image gets assigned to the 2jth row of new_image
            It doesn't clear the picture simply.
            say
                f.shape = (5,5)
                according to the transformation the shape of g has to be (5,10)
                which shows that the image has be scaled in X direction.
            But it is clearly visible that some of the columns may get untouched in g. To solve this issue we use a technique called
            'Interpolation'.
            InterPolation means, filling up the pixel with data available in neighbouring pixels.

    Translation:Shifting each pixel in Either or both direction
        g[i,j+x]   = f[i,j]
    or, g[i+y,j]   = f[i,j]
    or, g[i+y,j+x] = f[i,j]
        
        Empty Pixels are marked 0

    Rotation: Rotation simply means rotating an image around its centre of origin θ degrees

    All these Transformation can be done with the help of AFFINE TRANSFORMATION
    say,
        x' = a*x + tx
        y' = d*y + ty

        x'      a   0       x        tx
      |   | = |        |  |   |  + |     |
        y'      0   b       y        ty
"""
import matplotlib.pyplot as plt
import numpy as np

# Scaling Using PIL
from PIL import Image

im = Image.open('img.png')
new_widht = 512
new_height =512
new_im = im.resize((new_widht,new_height))
plt.imshow(new_im)
plt.show()

#rotating using PIL

theta = 45
rotated_im = im.rotate(theta)

plt.imshow(rotated_im)
plt.show()

#Scaling using cv2
import cv2

im_cv2 = cv2.imread('img.png',cv2.IMREAD_COLOR)
im_cv2 = cv2.cvtColor(im_cv2,cv2.COLOR_BGR2RGB)
scaled_im_cv2 = cv2.resize(im_cv2,None,fx=3,fy=1,interpolation=cv2.INTER_CUBIC)
plt.imshow(scaled_im_cv2)
plt.show()

#Translation in CV2 requires affine matrix M
rows,cols,_ = im_cv2.shape
# shape = (rows,cols,channels)
tx = 50
ty = 0
M = np.float32([[1 ,0, tx],[0 ,1, ty]])
translated_im_cv2 = cv2.warpAffine(im_cv2,M,(rows,cols))

plt.imshow(translated_im_cv2)
plt.show()

# Rotating an image using CV2
M = cv2.getRotationMatrix2D(center=(rows//2-1,cols//2-1),angle=theta,scale =1)
rotated_im_cv2 = cv2.warpAffine(im_cv2,M,(rows,cols))

plt.imshow(rotated_im_cv2)
plt.show()