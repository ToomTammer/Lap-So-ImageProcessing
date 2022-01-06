import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# loading image
img = cv.imread('tiny.png',)

# converting to gray scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# remove noise
img = cv.GaussianBlur(gray,(3,3),0)

# convolute with proper kernels
#แปลงภาพเป็นการไล่ระดับสีเราใช้cv2.CV_64F
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)  # x
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)  # y


#สีสามารถกำหนดด้วยคีย์เวิร์ด colors ใน cmap
plt.subplot(221),plt.imshow(img,cmap = 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(222),plt.imshow(laplacian,cmap = 'gray'),plt.title('Laplacian')
plt.xticks([]), plt.yticks([])

plt.subplot(223),plt.imshow(sobelx,cmap = 'gray'), plt.title('Sobel X') 
plt.xticks([]), plt.yticks([])

plt.subplot(224),plt.imshow(sobely,cmap = 'gray'), plt.title('Sobel Y')
plt.xticks([]), plt.yticks([])

plt.show()