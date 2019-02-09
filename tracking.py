#/bin/python3

# 20190209

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
np.set_printoptions(linewidth=np.inf)

## Constant parameters
fps = 0.6
window = 35

def edgeScanner(img_src):
    ret, img_binary = cv.threshold(img_src,1, 1, cv.THRESH_BINARY)
    element = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    edge = img_binary - cv.erode(img_binary, element)
    edge_point = np.where(edge == 1)
    return edge_point
    #TODO: error handle and connectivity proof

# choose worm 
img_first = cv.imread(img_dirs+'/'+files[0], -1)
worm_first = cv.selectROI(img_first, False)
ret, img_first_th= cv.threshold(img_first, 16000, 40000, cv.THRESH_TOZERO)
cv.rectangle(img_first, (worm_first[0], worm_first[1]),
    (worm_first[0]+worm_first[2], worm_first[1]+worm_first[3]),
    (0, 0, 0), 2 , 1)
img_first_th_roi = img_first_th[worm_first[1]:worm_first[1]+worm_first[3]+1,
    worm_first[0]:worm_first[0]+worm_first[2]]
y_ori = worm_first[1]
x_ori = worm_first[0]
pos_wormbody = np.where( img_first_th_roi > 0) 
pos_wormbody = list(zip(pos_wormbody[0] + y_ori, pos_wormbody[1] + x_ori))
pos_wormedge = edgeScanner(img_first_th_roi)
x_mid = max(pos_wormedge[1]) + min(pos_wormedge[1]) + x_ori
y_mid = max(pos_wormedge[0]) + min(pos_wormedge[0]) + y_ori
l = int(window/2)
pos_wormedge = list(zip(pos_wormedge[0] + y_ori, pos_wormedge[1] + x_ori))
cv.imwrite(out_dirs + '/'+ files[0] +'choose_roi.tif', img_first);

img_wormbody = img_first_th[y_mid-l-1:y_mid+l+1, x_mid-l-1:x_mid+l+1]

worms_t = {}
worm_t = '' # worm_t time series: (t, y_0, x_0, img_wormbody)
worm_t = [(0, y_mid-l-1, x_mid-l-1, img_wormbody)]
worms_t[0] = worm_t

## find the connected edge to represente worm edge
counter = 1
worm = 0
for img_name in files:
    # from previous frame
    y_ori = worms_t[worm][counter-1][1]
    x_ori = worms_t[worm][counter-1][2]
    img_wormbody = worms_t[worm][counter-1][3]
    # from image to pos
    #pos_wormbody = np.where( img_worm_p > 0) 
    #pos_wormbody = list(zip(pos_wormbody[1] + y_ori, pos_wormbody[0] + x_ori))
    #pos_wormedge = edgeScanner(img_worm_p)
    #pos_wormedge = list(zip(pos_wormedge[1] + y_ori, pos_wormedge[0] + x_ori))

    img_new = cv.imread(img_dirs+'/'+img_name, -1)

    img_new_roi = img_new[y_ori : y_ori+img_wormbody.shape[1]+1,
        x_ori : x_ori+img_wormbody.shape[0]+1]
    ret, img_new_roi_th= cv.threshold(img_new_roi, 16000, 40000, cv.THRESH_TOZERO)
    pos_wormedge_new = pos_wormedge
    pos_wormbody_new = pos_wormbody # It is necessary to regenerate position
    while(True): # Extend and erode edge step by step
        for point in pos_wormedge:
            print('Edge:',point)
            pos_x = point[1]
            pos_y = point[0]
            neighbour_set = [(pos_y, pos_x-1), (pos_y, pos_x+1), 
                (pos_y-1, pos_x), (pos_y+1, pos_x)]
            if img_new_roi_th[pos_y-y_ori, pos_x-x_ori]:
            # edge point still inside worm body 
                for neighbour in neighbour_set: # check each neighbour
                    if ((not(neighbour in pos_wormbody)) and # outside wormbody
                        (not(neighbour in pos_wormedge)) and #no orignal wormedge
                        img_new_roi_th[neighbour[0]-y_ori, neighbour[1]-x_ori]):
                        # extend edge when it is light
                        pos_wormbody_new.append(neighbour)
                        pos_wormedge_new.remove(point)
                        pos_wormedge_new.append(neighbour)
                        break
            else: # edge point already moved from wormbody
                for neighbour in neighbour_set:
                    if ((not(neighbour in pos_wormedge)) and
                        (neighbour in pos_wormbody)): # erode edge when it is dark
                        pos_wormbody_new.remove(point)
                        pos_wormedge_new.remove(point)
                        pos_wormedge_new.append(neighbour)
                        break
        if(pos_wormedge == pos_wormedge_new):
            break # until worm's edge is fixed
        pos_wormedge = pos_wormedge_new
        pos_wormbody = pos_wormbody_new
    minx, maxx = pos_wormedge[0]
    miny, maxy = pos_wormedge[0]
    for i in pos_wormedge:
        if minx > i[1]:
            minx = i[1]
        if miny > i[0]:
            miny = i[0]
        if maxx < i[1]:
            maxx = i[1]
        if maxy < i[0]:
            maxy = i[0]
                        
    #TODO: pos to image
    if (maxx-minx > 0) and (maxy-miny > 0):
        p1 = (x_ori, y_ori)
        p2 = (x_ori+img_new_roi.shape[1], y_ori+img_new_roi.shape[0])
        cv.rectangle(img_new, p1, p2, (65536, 65536, 65536), 1 ,cv.LINE_4)
        #img_wormbody = img_new_roi_th
        #img_wormbody = img_new_roi_th[miny-y_ori_new-7: maxy-y_ori_new+1+7,
        #    minx-x_ori_new-7: maxx-x_ori_new+1+7]
        #img_wormbody_binary = cv.threshold(img_wormbody,1, 1, cv.THRESH_BINARY)
        #element = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        #edge = img_wormbody_binary[1] - cv.erode(img_wormbody_binary[1], element)
        #edge = 1 - edge
        body = np.zeros(img_wormbody.shape)
        for i in pos_wormbody:
            body[i[0]-y_ori, i[1]-x_ori] = 1 
        x_mid = int((maxx+minx)/2)
        y_mid = int((maxy+miny)/2)
        l = int(window/2)
        img_wormbody = img_new[y_mid-l-1:y_mid+l+1, x_mid-l-1:x_mid+l+1]*body
        #edge = np.ones(img_wormbody.shape)
        for i in pos_wormedge:
            img_new[i] = 0 
        #img_new[y_mid-l-1 : y_mid+l+1, x_mid-l-1:x_mid+l+1]=img_new[
        #    y_mid-l-1:y_mid+l+1, x_mid-l-1:x_mid+l+1]*edge
        cv.imwrite('singleworm/'+ img_name+'.tif', img_wormbody)
    else :
        cv.putText(img, "Tracking fail", (100,80,), cv.FONT_HERSHEY_SIMPLEX,
            0.75,(0,0,255),2)
    worm_t.append((counter, y_mid-l-1, x_mid-l-1, img_wormbody))
    worms_t[worm] = worm_t
    cv.imwrite(out_dirs + '/'+ img_name +'.tif', img_new)

    counter = counter  + 1
    print("Frame: ", counter)
