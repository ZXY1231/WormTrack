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
#tracker = cv.TrackerMIL_create()
#tracker = cv.TrackerBoosting_create()
first_frame = cv.imread(img_dirs+'/'+files[0], -1)
#first_frame = ((first_frame-10000)/117).astype('uint8')
#ret,first_frame = cv.threshold(first_frame, 18000,40000,cv.THRESH_TOZERO)
box = cv.selectROI(first_frame, False)
bbox = [box[0], box[1], box[2], box[3]]
cv.rectangle(first_frame, (box[0], box[1]),
    (box[0]+box[2], box[1]+box[3]), (255, 255, 255), 2 , 1)
cv.imwrite(out_dirs + '/'+ files[0] +'choose_roi.tif', first_frame);#binary2-binary1)
#ok = tracker.init(first_frame, bbox)
worm = '' # worm grah: (t, UID, bbox, worm)
worm_set = '' # 
gap = 0 # gap bettwen worm & rectangle

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
    #cv.rectangle(img, (v_0,h_0), (v_1, h_1))
    #plt.savefig(out_dirs + '/'+ img_name +'.png', dpi=300)
    img_roi = img[int(bbox[0]-0.5*bbox[2]) : int(bbox[0] + 1.5*bbox[2]),
         int(bbox[1]-0.5*bbox[3]): int(bbox[1] + 1.5*bbox[3])]
    ret,img_thre = cv.threshold(img_roi, 15000, 40000, cv.THRESH_TOZERO)
    #index_worm = np.where(img_thre > 0)
    index_worm = np.where(img_thre == np.max(img_thre))
    cv.rectangle(img, 
        (int(bbox[0]-0.5*bbox[2]), int(bbox[1]-0.5*bbox[3])),
        (int(bbox[0] + 1.5*bbox[2]), int(bbox[1] + 1.5* bbox[3])),
        (0, 0, 0), 2 , 1)
    #bbox[0] = min(index_worm[0]) + int(bbox[0]-0.5*bbox[2])
    #bbox[1] = min(index_worm[1]) + int(bbox[1]-0.5*bbox[2])
    bbox[0] = index_worm[0] + int(0.5*bbox[2]) + int(bbox[0]-0.5*bbox[2])
    bbox[1] = index_worm[1] + int(0.5*bbox[3]) + int(bbox[1]-0.5*bbox[2])

    #bbox[2] = max(index_worm[0]) - min(index_worm[0])
    #bbox[3] = max(index_worm[1]) - min(index_worm[1])
    #img_worm = img_thre[bbox[0] : bbox[1], 
    #    (bbox[0] + bbox[2]) : (bbox[1] + bbox[3])]

    if (bbox[2] > 0) and (bbox[3] > 0):
        p1 = (bbox[0] , bbox[1])
        p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        cv.rectangle( img, p1, p2, (65536, 65536, 65536), 2 , 1)
    else :
        cv.putText(img, "Tracking fail", (100,80,), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    
    cv.imwrite(out_dirs + '/'+ img_name +'.tif', img);#binary2-binary1)
    #cv.imwrite(out_dirs + '/'+ img_name +'roi.tif', img_roi);#binary2-binary1)
    frame = frame + 1
    print("Frame: ", frame)
