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
# inital tracker
tracker = cv.TrackerMIL_create()
#tracker = cv.TrackerBoosting_create()
first_frame = cv.imread(img_dirs+'/'+files[0], -1)
ret,first_frame = cv.threshold(first_frame, 18000,40000,cv.THRESH_TOZERO)
bbox = cv.selectROI(first_frame, False)
ok = tracker.init(first_frame, bbox)

for img_name in files:
    img = cv.imread(img_dirs+'/'+img_name, -1)
    x_max = (img.shape)[0]
    y_max = (img.shape)[1]
    #img = ((img-10000)/117).astype('uint8')
    #img = img[int(1/3*x_max): int(2/3*x_max), int(1/3*y_max):int(2/3*y_max)]
    #edges = cv.Canny(img, 28, 143)
    #binary1 = cv.GaussianBlur(img, (3,3), 0);
    #binary2 = cv.GaussianBlur(img, (31,31), 0);
    #binary2 = cv.GaussianBlur(img, (11,11), 0);
    #plt.subplot(121), plt.imshow(img)
    #plt.title('Orignal Image')
    #plt.subplot(122), plt.imshow(binary1 - binary2)#binary1)
    #plt.subplot(122), plt.imshow(edges)
    #plt.title('Edge Image')
    #plt.show()
    #cv.rectangle(img, (v_0,h_0), (v_1, h_1))
    #plt.savefig(out_dirs + '/'+ img_name +'.png', dpi=300)
    ret,img = cv.threshold(img,18000,40000,cv.THRESH_TOZERO)
    ok, bbox = tracker.update(img)

    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle( img, p1, p2, (255, 255, 255), 2 , 1)
    else :
        cv.putText(img, "Tracking fail", (100,80,), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    
    cv.imwrite(out_dirs + '/'+ img_name +'boosting.tif', img);#binary2-binary1)
    frame = frame + 1
    print("Frame: ", frame)
