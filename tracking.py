#/bin/python3

# 20190208

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
args = parser.parse_args()
img_dirs = args.path
out_dirs = args.output
files = os.listdir(img_dirs)
files.sort()

## Constant parameters
fps = 0.6

# choose worm 
img_first = cv.imread(img_dirs+'/'+files[0], -1)
worm_first = cv.selectROI(img_first, False)
cv.rectangle(img_first, (worm_first[0], worm_first[1]),
    (worm_first[0]+worm_first[2], worm_first[1]+worm_first[3]),
    (0, 0, 0), 2 , 1)
cv.imwrite(out_dirs + '/'+ files[0] +'choose_roi.tif', first_frame);

worms_t = '' # worm_t time series: (t, x_0, y_0, img_wormbody)

## find the connected edge to represente worm edge
def edgeScanner(img_src):
    img_binary = cv.threshold(img_src,1, 1, cv.THRESH_BINARY)
    element = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    edge = img_binary - cv.erode(img_binary, element)
    edge_point = np.where(edge == 1)
    return edge_point
    #TODO: error handle and connectivity proof

for img_name in files:
    # from previous frame
    img_worm_p = worms_t[3]
    x_ori = worms_t[1]
    y_ori = worms_t[2]
    pos_wormbody = np.where( img_worm_p > 0) 
    pos_wormbody = list(zip(pos_wormbody[1] + x_ori, pos_wormbody[0] + y_ori))
    pos_wormedge = edgeScanner(img_worm_p)
    pos_wormedge = list(zip(pos_wormedge[1] + x_ori, pos_wormedge[0] + y_ori))

    img_new = cv.imread(img_dirs+'/'+img_name, -1)

    ret,img_thre = cv.threshold(img_roi, 15000, 40000, cv.THRESH_TOZERO)

    pos_wormedge_new = pos_wormedge
    pos_wormbody_new = pos_wormbody
    while(True):
        for point in pos_wormedge:
            neighbour_set = [(point[0], point[1]+1), (point[0], point[1]-1), 
                (point[0]-1, point[1]), (point[0]+1, point[1])]
            if img_new[point]: # edge poin still inside worm body 
                for neighbour in neighbour_set:
                    if (not(neighbour in pos_wormbody)) and
                        (neighbour in pos_wormedge) and
                        img_new(neighbour): # extend edge when it is light
                        pos_wormbody_new.appoend(neighbour)
                        pos_wormedge_new.remove(neighbour)
                        pos_wormedge_new.append(neighbour)
                        break
            else: # edge point already moved from wormbody
                for neighbour in neighbour_set:
                    if (not(neighbour in pos_wormedge)) and
                        (neighbour in pos_wormbody) # erode edge when it is dark
                        pos_wormbody_new.remove(point)
                        pos_wormedge_new.remove(point)
                        pos_wormedge_new.append(neighbour)
                        break
        if(pos_wormedge == pos_wormedge_new):
            break # until worm's edge is fixed
        pos_wormedge = pos_wormedge_new
        pos_wormbody = pos_wormbody_new
    minx, maxx = pos_wormedge[0][0]
    miny, maxy = pos_wormedge[0][1]
    for i in pos_wormedge:
        if minx > i[0]:
            minx = i[0]
        if miny > i[1]:
            miny = i[1]
        if maxx < i[0]:
            maxx = i[0]
        if maxy < i[1]:
            maxy = i[1]
                        

    #TODO: pos to image
    if (maxx-minx > 0) and (maxy-miny > 0):
        p1 = (minx, miny)
        p2 = (maxx, maxy)
        cv.rectangle( img, p1, p2, (65536, 65536, 65536), 1 ,cv.LINE_4)
    else :
        cv.putText(img, "Tracking fail", (100,80,), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    
    worm_t.append((time, minx, miny, img_wormbody))
    cv.imwrite(out_dirs + '/'+ img_name +'.tif', img);

    frame = frame + 1
    print("Frame: ")
