# to open/load image
from PIL import Image,ImageFilter
# to render image
import matplotlib.pyplot as plt
# to make kernels and noise matrix
import numpy as np

# A helper function to show images side by

def show_side_by_side(im1,title1,im2,title2,gray=False):
    CMAP = 'viridis' if not gray else 'gray'
    plt.subplot(1,2,1)
    plt.title(title1)
    plt.imshow(im1,cmap=CMAP)

    plt.subplot(1,2,2)
    plt.title(title2)
    plt.imshow(im2,cmap=CMAP)
    
    plt.show()

im = Image.open('lenna.png')
rows,cols = im.size
# Okay now lets create some noise
noise = np.random.normal(0,15,size=(rows,cols,3)).astype(np.uint8)

noised_im_arr = im + noise

plt.figure(figsize=(7,7))

show_side_by_side(im,'Original',noised_im_arr,'Noised Image')

# Now lets filter the noise
# mean filtering
noised_im = Image.fromarray(noised_im_arr)

kernel = np.ones((5,5))/36
kernel_filter = ImageFilter.Kernel((5,5),kernel.flatten())

filtered_img = noised_im.filter(kernel_filter)

show_side_by_side(noised_im,'Noised Image',filtered_img,'Filtered Image')

# Smaller kernel can make a sharp image but trade off the noise reduction capabilities
small_kernel = np.ones((3,3))/9
small_kernel_filte = ImageFilter.Kernel((3,3),small_kernel.flatten())

filtered_img_small_kernel = noised_im.filter(small_kernel_filte)

show_side_by_side(filtered_img,'5 x 5 kernel',filtered_img_small_kernel,'3 x 3 kernel')
# It is clearly visible that her shoulder is now sharper but the green spots are brighter too

# Noise reduction can be achieved with Gaussian Blur too
# ImageFilter.GaussianBlur
# Default Radius is 2

gaussian_img = noised_im.filter(ImageFilter.GaussianBlur)
show_side_by_side(noised_im,'Noised Image',gaussian_img,'Gaussian Blurred Img')

# Sharpening Images
# For image sharpening we will use commonly used kernel
#           -1 -1 -1
#           -1  9 -1
#           -1 -1 -1

sharpen_kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])

sharpen_filter = ImageFilter.Kernel((3,3),sharpen_kernel.flatten())

sharped_gaussian = gaussian_img.filter(sharpen_filter)
sharpen_img = im.filter(sharpen_filter)
show_side_by_side(gaussian_img,'Gaussian blurred Img',sharped_gaussian,'Sharpened Gaussian Img')
show_side_by_side(im,'orginal image',sharpen_img,'Sharpened Image')

# There's a SHARPEN filter provided by PIL by default
PIL_sharpen_img = im.filter(ImageFilter.SHARPEN)
show_side_by_side(sharpen_img,'Sharpen by manual',PIL_sharpen_img,'Sharpen by PIL')

#   Edge Detection
# Before Finding Edge Use ImageFilter.ENHANCE_EDGE filter to enhance the edges of an images so that you can find more and more edges
img = Image.open('barbara.png')
enhanced_egde = img.filter(ImageFilter.EDGE_ENHANCE)

show_side_by_side(img,'Original',enhanced_egde,'Enhanced Edges',gray=True)

# Now we have enhanced edges we can start finding the edges in the image

find_edge = img.filter(ImageFilter.FIND_EDGES)
show_side_by_side(img,'Original Image',find_edge,'Edges in the Image',gray=True)

#   Median Filter
#   Median Filter takes the median of the neighbouring pixel and replace the middle one with median

im2 = Image.open('cameraman.jpeg')

median_filtered = im2.filter(ImageFilter.MedianFilter)
# median filters are good for increasing the segmentation within the images
show_side_by_side(im2,'Original',median_filtered,'Median Filtered')
