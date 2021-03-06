#/bin/python3

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import sys
import argparse


## Pass through all parameters
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('output', type=str)
#parser.add_argument('name', type=str)
args = parser.parse_args()
img_dirs = args.path
out_dirs = args.output
#name = args.name

files = os.listdir(img_dirs)
files.sort()

## Constant number
fps = 0.6

# Local the neurons

# extra fluorescence
frame = 0

for img_name in files:
    img = cv.imread(img_dirs+'/'+img_name, 2)
    x_max = (img.shape)[0]
    y_max = (img.shape)[1]
    #img = ((img-10000)/117).astype('uint8')
    #img = img[int(1/3*x_max): int(2/3*x_max), int(1/3*y_max):int(2/3*y_max)]
    #edges = cv.Canny(img, 28, 143)
    binary1 = cv.GaussianBlur(img, (3,3), 0);
    #binary2 = cv.GaussianBlur(img, (31,31), 0);
    binary2 = cv.GaussianBlur(img, (11,11), 0);
    plt.subplot(121), plt.imshow(img)
    plt.title('Orignal Image')
    plt.subplot(122), plt.imshow(binary1 - binary2)#binary1)
    #plt.subplot(122), plt.imshow(edges)
    plt.title('Edge Image')
    #plt.show()
    #cv.rectangle(img, (v_0,h_0), (v_1, h_1))
    #plt.savefig(out_dirs + '/'+ img_name +'.png', dpi=300)
    cv.imwrite(out_dirs + '/'+ img_name +'.tif', binary2-binary1)
    frame = frame + 1
    print("Frame: ", frame)
