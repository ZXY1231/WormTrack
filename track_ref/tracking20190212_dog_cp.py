#/bin/python3

# 20190212

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import sys
import argparse
import csv

## Pass through all parameters
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()
img_dirs = args.path
out_dirs = args.output
files = os.listdir(img_dirs)
files.sort()
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.nan)
## Constant parameters
fps = 0.6
window = 35

def extendBody( img_new_roi_th_l, img_wormbody_old_l):
    A = cv.threshold(img_new_roi_th_l, 10000, 1, cv.THRESH_BINARY)[1]
    element = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    #A = cv.erode(A, element)
    img_wormbody_old_l = cv.erode(img_wormbody_old_l, element)
    X1 = cv.threshold(img_wormbody_old_l, 0, 1, cv.THRESH_BINARY)[1] * A
    X0 = np.zeros(X1.shape, dtype=np.uint16)
    while(sum(sum( X0 - X1))):
        X0 = X1
        #X1 = cv.bitwise_and(cv.dilate(X0, element), A)
        X1 = (cv.dilate(X0, element))*A
    img_wormbody_new_l = X1
    return img_wormbody_new_l

def edgeScanner(img_src):
    #ret, img_binary = cv.threshold(img_src,1, 1, cv.THRESH_BINARY)
    ret, img_binary = cv.threshold(img_src, 0, 1, cv.THRESH_BINARY)
    #ret, img_binary = cv.threshold(img_src,50000, 1, cv.THRESH_BINARY)
    element = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    edge = img_binary - cv.erode(img_binary, element)
    edge_point = np.where(edge == 1)
    return edge_point
    #TODO: error handle and connectivity proof

# Choose worm 
img_first = cv.imread(img_dirs+'/'+files[0], -1)
worm_first = cv.selectROI(img_first, False)
#ret, img_first_th= cv.threshold(img_first, 16000, 40000, cv.THRESH_TOZERO)
img_first_roi = img_first[worm_first[1]:worm_first[1]+worm_first[3]+1,
    worm_first[0]:worm_first[0]+worm_first[2]]
img_first_roi_th = (cv.GaussianBlur(img_first_roi, (31,31), 0)
                    - cv.GaussianBlur(img_first_roi, (3,3), 0))

y_ori = worm_first[1]
x_ori = worm_first[0]
pos_wormbody = np.where( img_first_roi_th > 10000)
x_mid = int((max(pos_wormbody[1]) + min(pos_wormbody[1]))/2 + x_ori)
y_mid = int((max(pos_wormbody[0]) + min(pos_wormbody[0]))/2 + y_ori)
pos_wormbody = list(zip(pos_wormbody[0] + y_ori, pos_wormbody[1] + x_ori))
l = int(window/2)

img_wormbody = img_first[y_mid-l:y_mid+l+1, x_mid-l:x_mid+l+1]
body_mask = np.zeros(img_wormbody.shape, dtype=np.uint16)
for i in pos_wormbody:
    body_mask[i[0]-y_mid+l, i[1]-x_mid+l] = 1
img_wormbody = img_wormbody * body_mask

cv.rectangle(img_first, (worm_first[0], worm_first[1]),
    (worm_first[0]+worm_first[2], worm_first[1]+worm_first[3]),
    (0, 0, 0), 1 , 1)
cv.imwrite(out_dirs + '/'+ files[0] +'choose_roi.tif', img_first)

worms_t = {}
worm_t = '' # worm_t time series: (t, y_0, x_0, img_wormbody)
worm_t = [(0, y_mid-l, x_mid-l, img_wormbody, img_first_roi_th)]
worms_t[0] = worm_t

# CSV file:
header = ['worm', 'time', 'y_0', 'x_0', 'img_wrombody', 'img_new_roi_th']
rows = [0, 0, y_mid-l, x_mid-l, files[0]]
csvfile =  open(out_dirs+'.csv', 'w')
spamwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
spamwriter.writerow(header)
spamwriter.writerow(rows)

## find the connected edge to represente worm edge
counter = 1
worm = 0
y_ori_new = ''
x_ori_new = ''
for img_name in files:
    # from previous frame
    y_ori = worms_t[worm][counter-1][1]
    x_ori = worms_t[worm][counter-1][2]
    img_wormbody_old = worms_t[worm][counter-1][3]

    img_new = cv.imread(img_dirs+'/'+img_name, -1)
    img_new_roi = img_new[y_ori : y_ori+img_wormbody.shape[0],
        x_ori : x_ori+img_wormbody.shape[1]]
    #ret, img_new_roi_th= cv.threshold(img_new_roi, 16000, 40000, cv.THRESH_TOZERO)
    img_new_roi_th = (cv.GaussianBlur(img_new_roi, (31,31), 0)
                      -cv.GaussianBlur(img_new_roi, (3,3), 0))
    img_wormbody = extendBody(img_new_roi_th, img_wormbody_old)
    pos_wormbody = np.where( img_wormbody > 0) 
    pos_wormbody = list(zip(pos_wormbody[0] + y_ori, pos_wormbody[1] + x_ori))
    pos_wormedge = edgeScanner(img_wormbody)
    pos_wormedge = list(zip(pos_wormedge[0] + y_ori, pos_wormedge[1] + x_ori))
    
    # Find boundary retangle
    miny, minx = pos_wormedge[0]
    maxy, maxx = pos_wormedge[0]
    for i in pos_wormedge:
        if minx > i[1]:
            minx = i[1]
        if miny > i[0]:
            miny = i[0]
        if maxx < i[1]:
            maxx = i[1]
        if maxy < i[0]:
            maxy = i[0]
                        
    if (maxx-minx > 0) and (maxy-miny > 0): # to avoid unacceptable result
        p1 = (x_ori, y_ori)
        p2 = (x_ori+img_new_roi.shape[1], y_ori+img_new_roi.shape[0])

        # Recenter wormbody
        x_mid = int((maxx+minx)/2)
        y_mid = int((maxy+miny)/2)
        l = int(window/2)
        x_ori_new = x_mid - l
        y_ori_new = y_mid - l

        img_wormbody_mask = np.zeros(img_wormbody.shape, dtype=np.uint16)
        for i in pos_wormbody:
            img_wormbody_mask[i[0]-y_ori_new, i[1]-x_ori_new] = 1 
        img_wormbody = (img_new[y_ori_new : y_mid+l+1, x_ori_new : x_mid+l+1]
                        * img_wormbody_mask)
        # draw egde in orignal imagine
        for i in pos_wormedge:
            img_new[i] = 0 
        cv.imwrite('singleworm/'+ img_name+'.tif', img_wormbody)
        cv.rectangle(img_new, p1, p2, (65536, 65536, 65536), 1 ,cv.LINE_4)
    else :
        cv.putText(img_new, "Tracking fail", (100,80,), cv.FONT_HERSHEY_SIMPLEX,
            0.75,(0,0,255),2)
    worm_t.append((counter, y_ori_new, x_ori_new, img_wormbody, img_new_roi_th))
    worms_t[worm] = worm_t
    rows = [0, counter, y_ori_new, x_ori_new, img_name]
    spamwriter.writerow(rows)
    cv.imwrite(out_dirs + '/'+ img_name +'.tif', img_new)

    counter = counter  + 1
    print("Frame: ", counter)

csvfile.close()
