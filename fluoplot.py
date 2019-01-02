#/bin/python3

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('RGECO-Diacyle-line3_5/RGECO-Diacyle-line3_500000000.tif',0)
print(img)
'''
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
'''

alpha = 1
beta = 0
new_image =  alpha*img + beta


cv.imshow('image', img)
cv.imshow('new_image', new_image)
cv.waitKey(0)
cv.destroyAllWindows()

#cv.imwrite('output.png', img)
