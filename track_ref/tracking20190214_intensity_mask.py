#/bin/python3

# 20190213

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import sys
import argparse
import csv
import time

# TODO: add extral edge outside whole image to avoid out of boundary

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
fps = 1/2
window = 35
log_time = time.strftime("%m%d%H%M", time.localtime())
worms_t = {}
worm_t = {} # worm_t time series: (t, y_0, x_0, img_wormbody)
worm = 0
# TODO: add range handle

# select points which intensity between 1/2~3/4 order
def intensityMask(img_wormbody_l):
    intensity_mask = np.ones(img_wormbody_l.shape)
    img_wormbody_l_ravel = (np.copy(img_wormbody_l)).ravel()
    img_wormbody_l_ravel.sort()
    pos_nonzero = img_wormbody_l_ravel.nonzero()[0]
    pos_lowboundary = 0 
    pos_upboundary  = len(pos_nonzero) - 1
    if(len(pos_nonzero) < 20):
        print("Too less wormbody pixel")
    else:
        pos_lowboundary = len(pos_nonzero) - 1 - len(pos_nonzero)//5
        pos_upboundary = len(pos_nonzero) - 1 - 3 # take out highest 3
    intensity_lowboundary = img_wormbody_l_ravel[ pos_nonzero[pos_lowboundary]]
    intensity_upboundary = img_wormbody_l_ravel[ pos_nonzero[pos_upboundary]]
    intensity_mask = cv.threshold(img_wormbody_l, intensity_upboundary, 65535,
            cv.THRESH_TOZERO_INV)[1]
    intensity_mask = cv.threshold(intensity_mask, intensity_lowboundary, 1,
            cv.THRESH_BINARY)[1]
    return intensity_mask

# X_k = (X_{k-1} (+) B ) ^ A
def extendBody( img_new_roi_th_l, img_wormbody_old_l, counter_l, worm_l):
    A = cv.threshold(img_new_roi_th_l, 10000, 1, cv.THRESH_BINARY)[1]
    img_wormbody_old_l_bi = cv.threshold(img_wormbody_old_l, 0, 1,
            cv.THRESH_BINARY)[1]
            
    # limit search window's boundary away from original edge 3 pixels
    A_mask_p1 = cv.dilate(img_wormbody_old_l_bi, np.ones([7,7]))
    A_mask_p2 = np.ones(A.shape)
    '''
    if counter_l > 4:
        img_p2 = worms_t[worm_l][counter_l-2][3]
        img_p2 = cv.threshold(img_p2, 0, 1, cv.THRESH_BINARY)[1]
        pos_p2_y = worms_t[worm_l][counter_l-2][1]
        pos_p2_x = worms_t[worm_l][counter_l-2][2]
        pos_p1_y = worms_t[worm_l][counter_l-1][1]
        pos_p1_x = worms_t[worm_l][counter_l-1][2]
        img_p2 = cv.dilate(img_p2, np.ones([7,7]))
        # coordation transform
        A_mask_p2= np.roll(img_p2, pos_p2_y - pos_p1_y, axis = 0)
        A_mask_p2= np.roll(A_mask_p2, pos_p2_x - pos_p1_x, axis = 1)
    '''
    X1_mask_intensity = intensityMask(img_wormbody_old_l)
    
    A = A * A_mask_p1 * A_mask_p2 
    element = np.ones([3,3])
    X1 = cv.erode(img_wormbody_old_l_bi, element) *X1_mask_intensity* A 
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


# Choose the inital worm population
def chooseWorm():
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
    worm_t[0] = (0, y_mid-l, x_mid-l, img_wormbody, img_first_roi_th)
    worms_t[0] = worm_t
    rows = [0, 0, y_mid-l, x_mid-l, files[0]]
    spamwriter.writerow(rows)

# CSV file:
header = ['worm', 'time', 'y_0', 'x_0', 'img_wrombody', 'img_new_roi_th']
csvfile =  open(out_dirs+ '/' + os.path.basename(img_dirs)+log_time+'.csv', "w+")
spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
spamwriter.writerow(header)
#Maybe cause error
out_dirs_img  = out_dirs +'/'+os.path.basename(img_dirs)+'-'+log_time 
out_dirs_sw = out_dirs +'/'+os.path.basename(img_dirs)+'-'+log_time + "-%d"%worm
if not os.path.isdir(out_dirs_img):
    os.mkdir(out_dirs_img)
if not os.path.isdir(out_dirs_sw):
    os.mkdir(out_dirs_sw)



## find the connected edge to represente worm edge
counter = 1
worm = 0
y_ori_new = ''
x_ori_new = ''

chooseWorm()

for img_name in files:
    # from previous frame
    y_ori = worms_t[worm][counter-1][1]
    x_ori = worms_t[worm][counter-1][2]
    img_wormbody_old = worms_t[worm][counter-1][3]

    img_new = cv.imread(img_dirs+'/'+img_name, -1)
    img_new_roi = img_new[y_ori : y_ori+img_wormbody_old.shape[0],
        x_ori : x_ori+img_wormbody_old.shape[1]]

    #TODO: try to use fastNlMeansDenoising to denoise
    #ret, img_new_roi_th= cv.threshold(img_new_roi, 16000, 40000, cv.THRESH_TOZERO)
    img_new_roi_th = (cv.GaussianBlur(img_new_roi, (31,31), 0)
                      -cv.GaussianBlur(img_new_roi, (3,3), 0))
    img_wormbody = extendBody(img_new_roi_th, img_wormbody_old, counter, worm)
    pos_wormbody = np.where( img_wormbody > 0)
    # Find bounday, Recenter wormbody
    l = int(window/2)
    
    y_mid = (min(pos_wormbody[0] + max(pos_wormbody[0])))//2 + y_ori
    x_mid = (min(pos_wormbody[1] + max(pos_wormbody[1])))//2 + x_ori
    y_ori_new =  y_mid - l 
    x_ori_new =  x_mid - l 
    pos_wormbody = list(zip(pos_wormbody[0] + y_ori, pos_wormbody[1] + x_ori))
    pos_wormedge = edgeScanner(img_wormbody)
    pos_wormedge = list(zip(pos_wormedge[0] + y_ori, pos_wormedge[1] + x_ori))
    
    if (len(pos_wormbody[0]) > 1): # to avoid unacceptable result
        p1 = (x_ori, y_ori)
        p2 = (x_ori+img_new_roi.shape[1], y_ori+img_new_roi.shape[0])

        img_wormbody_mask = np.zeros(img_wormbody.shape, dtype=np.uint16)
        for i in pos_wormbody:
            img_wormbody_mask[i[0]-y_ori_new, i[1]-x_ori_new] = 1 
        img_wormbody = (img_new[y_ori_new : y_mid+l+1, x_ori_new : x_mid+l+1]
                        * img_wormbody_mask)
        # draw egde in orignal imagine
        for i in pos_wormedge:
            img_new[i] = 0 
        cv.imwrite(out_dirs_sw + '/' + img_name , img_wormbody)
        cv.rectangle(img_new, p1, p2, (65536, 65536, 65536), 1, cv.LINE_4)
    else :
        cv.putText(img_new, "Tracking fail", (100,80,), cv.FONT_HERSHEY_SIMPLEX,
            0.75,(0,0,255),2)
        print("Track Fail")
    worm_t[counter] = (counter, y_ori_new, x_ori_new, img_wormbody, img_new_roi_th)
    worms_t[worm] = worm_t
    rows = [worm, counter*fps, y_ori_new, x_ori_new, img_name]
    spamwriter.writerow(rows)
    cv.imwrite(out_dirs_img + '/'+ img_name , img_new)

    counter = counter  + 1
    print("Frame: ", counter)

csvfile.close()
