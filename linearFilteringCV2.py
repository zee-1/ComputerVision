# for performing operations and loading Images 
import cv2

# for rendering images
import matplotlib.pyplot as plt

#for making kernels and matrices
import numpy as np

# A helper function to show images side by

def show_side_by_side(im1,title1,im2,title2,gray=False):
    CMAP = 'viridis' if not gray else 'gray'
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.title(title1)
    plt.imshow(im1,cmap=CMAP)

    plt.subplot(1,2,2)
    plt.title(title2)
    plt.imshow(im2,cmap=CMAP)
    
    plt.show()

im  = cv2.imread('lenna.png',cv2.IMREAD_COLOR)
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
rows,cols,_ = im.shape

# Now lets create some noise 
noise = np.random.normal(0,15,(rows,cols,_)).astype(np.uint8)

noised_im = im + noise

show_side_by_side(im,'Original',noised_im,'Noised Image')

# Now lets create a mean filter

kernel_mean = np.ones((6,6))/36

# Applying filter 
# cv2.filter2D(source matrix/Image, ddepth: depth of image, kernel)
mean_filtered_im = cv2.filter2D(noised_im,-1,kernel_mean)

show_side_by_side(noised_im,'Noised img',mean_filtered_im,'Mean filtered')

# Smaller kernels gives lesser blur but fails to remove noise better

smaller_kernel_mean = np.ones((4,4))/16
small_mean_filtered_img = cv2.filter2D(noised_im,-1,smaller_kernel_mean)

show_side_by_side(mean_filtered_im,'Mean filtered 6 x 6',small_mean_filtered_img,' Mean filtered 4 x 4')

# Applying Gaussian blur to remove noise and smoothen the image
# cv2.GaussianBlur(source image, ksize: kernel size, standard daviation along x, standard deviation along y)

gaussian_im = cv2.GaussianBlur(noised_im,ksize=(5,5),sigmaX=4,sigmaY=4)

show_side_by_side(noised_im,'Original Image',gaussian_im,'Gaussian Blured image')

# Sharpening of image
## Sharpening Images
# For image sharpening we will use commonly used kernel
#           -1 -1 -1
#           -1  9 -1
#           -1 -1 -1


sharpen_kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
sharpen_im = cv2.filter2D(im,ddepth=-1,kernel=sharpen_kernel)
show_side_by_side(im,'Original',sharpen_im,'Sharpen Image')

# Edge Detection
"""
    To seperate the edges
    First,  Smoothen the edges to avoid the chance of distortion caused by noise
    
    Secondly,   Apply Sobel operator along X and obtain Gx
                Apply Sobel operator along Y and obtain Gy
                G = [Gx, Gy]
                |G| will give the intensity values
    
    Lastly,     Plotting |G| will give the edges of the image

Sobel Matrix/operator:
    Along x:
    1 0 -1
    2 0 -2
    1 0 -1

    Along Y:
     1  2  1
     0  0  0
    -1 -2 -1
"""
im2 = cv2.imread('barbara.png',cv2.IMREAD_GRAYSCALE)

# cv2.Sobel(src:Source Image Matrix,ddepth: ddepth of the output image,dx: order of derivative along X,dy: order of derivative along Y,ksize: kernel Size)
ddepth = cv2.CV_16S
grad_x = cv2.Sobel(im2,ddepth=ddepth,dx=1,dy=0,ksize=3)
grad_y = cv2.Sobel(im2,ddepth=ddepth,dx=0,dy=1,ksize=3)

# calculate the |Gx| and |Gy| for calculation of |G|
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

# Combine the abs_grad-x and abs_grad_y using addWeighted method in cv2
# cv2.addWeighted(martix1, alpha channel, matrix 2, beta channel, gamma channel)
abs_g = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
show_side_by_side(abs_grad_x,'|Gx|',abs_grad_y,'|Gy|',gray=True)
show_side_by_side(im2,'Original',abs_g,'Detected Edges',gray=True)

##   Median Filter
#   Median Filter takes the median of the neighbouring pixel and replace the middle one with median
# In OpenCV the median filter can be applied by medianBlur method
# cv2.median(src: Source image/matrix,ksize: Kernel size)

median_filtered_im = cv2.medianBlur(im,3)
show_side_by_side(im,'Original',median_filtered_im,'Median filtered image')

# Thresholding: Thresholding means seperating pixel about certain threshold value
# Returns ret which is the threshold used and outs which is the image
ret, outs = cv2.threshold(src = im, thresh = 0, maxval = 255, type = cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

# Make the image larger when it renders
plt.figure(figsize=(10,10))

# Render the image
plt.imshow(outs, cmap='gray')