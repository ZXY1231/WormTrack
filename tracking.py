#/bin/python3
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import sys
import argparse
import csv
import time

'''
| version | Commit
| 0.1     | h.f. @ 20190214
# TODO: add range handle
'''

## Pass through all parameters
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()
img_dirs = args.path
out_dirs = args.output

############################ Constant parameters ##############################
files = os.listdir(img_dirs)
files.sort()
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.nan)

fps = 2
window = 35
log_time = time.strftime("%m%d%H%M", time.localtime())
worms_t = {} # 
worm_t = {} # worm_t time series dict: (t, y_0, x_0, img_wormbody)
worm = 0
counter = 1 # 0th is used to mask roi, don't treated as effeice data
worm_a_m_s = (0, 0, 0)

################################ Modoule #######################################
# select points which intensity between 1/2~3/4 order
def intensityMask(img_wormbody_l):
    intensity_mask = np.ones(img_wormbody_l.shape, dtype=np.uint16)
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
    # Error handle when boundary is not acceptable: too low,high, no value
    intensity_mask = cv.threshold(img_wormbody_l, intensity_upboundary, 65535,
            cv.THRESH_TOZERO_INV)[1]
    intensity_mask = cv.threshold(intensity_mask, intensity_lowboundary, 1,
            cv.THRESH_BINARY)[1]
    # Error handler: no active point
    return intensity_mask

def wormbody_pre(worm_l, counter_l):
    area_p = -1
    mean_p = -1
    std_p = -1
    if 3<counter_l <= 15:
        worms_img = [] 
        worms_area = []
        for i in range(1, counter_l):
            worms_img.append( worms_t[worm_l][i][3])
            worms_area.append(cv.countNonZero(worms_t[worm_l][i][3]))
        while(len(worms_img) > 2 ): # At least 3 elements
            if max(worms_area) - min(worms_area) < 0.2*min(worms_area):
                area_p = np.mean(worms_area)
                mean_p_list = []
                std_p_list = []
                for img in worms_img:
                    mean_t, std_t = cv.meanStdDev(img, None, None,
                            (img>0).astype(np.uint8))
                    mean_p_list.append(mean_t)
                    std_p_list.append(std_t)
                mean_p = np.mean(mean_p_list)
                std_p  = np.mean(std_p_list)
                break
            else:
                if len(worms_img) > 0: # Be care here
                    min_area = min(worms_area)
                    max_area = max(worms_area)
                    if ((max_area) - np.mean(worms_area) < 
                            (np.mean(worms_area) - min_area)):
                        index = worms_area.index(min_area)
                        worms_area.remove(min_area)
                        worms_img.pop(index)
                    else: # Remove max or min
                        index = worms_area.index(max_area)
                        worms_area.remove(max_area)
                        worms_img.pop(index)
    else: # counter_l > 15: # Process after frame 15
        window_size = 15
        search_shift = 15
        left = counter_l  - 15
        right = counter_l -1
        worms_img = [] 
        worms_area = []

        min_left = left - 1 - window_size - search_shift
        if min_left < 1:
            min_left = 1

        while(left > min_left):
            worms_area.clear()
            worms_img.clear()
            for i in range(left, right+1):
                worms_img.append( worms_t[worm_l][i][3])
                worms_area.append(cv.countNonZero(worms_t[worm_l][i][3]))
            if max(worms_area) - min(worms_area)< 0.2*min(worms_area):
                area_p = np.mean(worms_area)
                mean_p_list = []
                std_p_list = []
                for img in worms_img:
                    mean_t, std_t = cv.meanStdDev(img, None, None,
                            (img>0).astype(np.uint8))
                    mean_p_list.append(mean_t)
                    std_p_list.append(std_t)
                mean_p = np.mean(mean_p_list)
                std_p  = np.mean(std_p_list)
                break
            else:
                left -= 1
                right-= 1
    return area_p, mean_p, std_p



def evalBody(img_wormbody_l, worm_l, counter_l, img_whole, x_ori, y_ori):
    img_wormbody_mask = np.ones(img_wormbody_l.shape, dtype=np.uint16)
    img_wormbody_l_bi = cv.threshold(img_wormbody_l, 0, 1, 
            cv.THRESH_BINARY)[1]
    connected = cv.connectedComponents(img_wormbody_l_bi.astype(np.uint8))
    previous = wormbody_pre(worm_l, counter_l)
    if previous > (0, 0, 0): # Update value only when it have valid value
        global worm_a_m_s
        worm_a_m_s = previous
        print(worm_a_m_s)
    area_pre = worm_a_m_s[0]
    mean_pre = worm_a_m_s[1]
    std_pre = worm_a_m_s[2]

    if connected[0] > 2: # only has one obejct
        #print("")
        #else: # more than 2 object
        l = window//2 # Diffusion
        x_ori_extend = x_ori - l
        y_ori_extend = y_ori - l
        img_roi_extend = img_whole[y_ori_extend : y_ori_extend + 4*l,
            x_ori_extend : x_ori_extend + 4*l]
        img_wormbody_extend = np.zeros(img_roi_extend.shape, dtype=np.uint16)
        img_wormbody_extend[l: 3*l+1, l:3*l+1] = img_wormbody_l
        img_wormbody_extend_new = extendBody(img_roi_extend, img_wormbody_extend,
            counter_l, worm_l)
        img_wormbody_extend_new_bi = cv.threshold(img_wormbody_extend_new, 0, 1,
            cv.THRESH_BINARY)[1]
        connected = cv.connectedComponents(img_wormbody_extend_new_bi.astype(
            np.uint8))

        flag1 = 0
        flag2 = 0
        min_mean_diff = 2**16
        min_std_diff = 2**16
        
        #area_pre = cv.countNonZero(worms_t[0][counter_l-15][3]) 
        #mean_pre, std_pre = cv.meanStdDev(worms_t[0][counter_l-15][3], mean, std,
        #        (worms_t[0][counter_l-15][3]>0).astype(np.uint8))
        for conp in range(1, connected[0]):
            conp_mask = (connected[1] == conp).astype(np.uint8)
            area_conp = cv.countNonZero(conp_mask)
            if (area_conp - area_pre ) < 0.3 * area_pre:
                mean, std = cv.meanStdDev(img_wormbody_extend_new, None, None,
                        conp_mask)
                if abs(mean - mean_pre) < min_mean_diff :
                    flag1 = conp
                    min_mean_diff = abs(mean - mean_pre)
                if abs(std - std_pre) < min_std_diff :
                    flag2 = conp
                    min_std_diff = abs(std - std_pre)
        if not (flag1 == 0):
            if flag1 == flag2:
                img_wormbody_mask = (connected[1] == flag1)[l:3*l+1,
                        l:3*l+1].astype(np.uint16)
            else:
                img_wormbody_mask = (connected[1] == flag1)[l:3*l+1,
                        l:3*l+1].astype(np.uint16)
    return img_wormbody_mask

# Extend wormbody by finding connected component: X_k = ( X_{k-1} ⊕ SE ) ^ A
# Resample
def extendBody( img_new_roi_l, img_wormbody_old_l, counter_l, worm_l, I=False):
    # Difference of Gaussian: erode a ring between wormbody
    A = (cv.GaussianBlur(img_new_roi_l, (31,31), 0)
                      -cv.GaussianBlur(img_new_roi_l, (3,3), 0))
    A = cv.threshold(A, 10000, 1, cv.THRESH_BINARY)[1]
    img_wormbody_old_l_bi = cv.threshold(img_wormbody_old_l, 0, 1,
        cv.THRESH_BINARY)[1]
        # Only search wormbody in this region
    if I :
        # Limit search window's boundary away from original edge 3 pixels
        A_mask_p1 = cv.dilate(img_wormbody_old_l_bi, np.ones([7,7]))
        A = A * A_mask_p1
    element = np.ones([3,3])
    # Initial points to start extend wormbody
    X1_mask_intensity = intensityMask(img_wormbody_old_l)
    seed = cv.erode(img_wormbody_old_l_bi, element) * X1_mask_intensity * A 
    X1 = seed
    X0 = np.zeros(X1.shape, dtype=np.uint16)
    while(sum(sum( X0 - X1))):
        X0 = X1
        X1 = (cv.dilate(X0, element)) * A
    img_wormbody_new_l = X1 * img_new_roi_l #* evalBody(X1, worm_l, counter_l)
    # TODO: when mask is empty
    return img_wormbody_new_l

## Return edge mask: edge_mask = binary(img_src) - [binary(img_src) ⊖ SE]
def edgeScanner(img_src):
    ret, img_binary = cv.threshold(img_src, 0, 1, cv.THRESH_BINARY)
    element = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    edge_mask = img_binary - cv.erode(img_binary, element)
    return edge_mask

# initialize worm population
# TODO: select mutli worm by manual or pass throught from mask image file
def initWormMask():
    img_first = cv.imread(img_dirs+'/'+files[0], -1) # Read as origal format
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
    mean = 0
    std = 0
    global worm_a_m_s
    mean, std = cv.meanStdDev(img_wormbody, mean, std, body_mask.astype(np.uint8))
    area = cv.countNonZero(img_wormbody)
    worm_a_m_s = (area, mean, std)
    worms_t[0] = worm_t
    rows = [0, 0, y_mid-l, x_mid-l, files[0]]
    spamwriter.writerow(rows)

############################ Main function #####################################
# Create CSV file to record info
header = ['worm', 'time', 'y_0', 'x_0', 'img_wrombody']
csvfile =  open(out_dirs+ '/' + os.path.basename(img_dirs)+log_time+'.csv', "w+")
spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
spamwriter.writerow(header)

# Create data folder
out_dirs_img  = out_dirs +'/'+os.path.basename(img_dirs)+'-'+log_time 
out_dirs_sw = out_dirs +'/'+os.path.basename(img_dirs)+'-'+log_time + "-%d"%worm
if not os.path.isdir(out_dirs_img):
    os.mkdir(out_dirs_img)
if not os.path.isdir(out_dirs_sw):
    os.mkdir(out_dirs_sw)


initWormMask()

for img_name in files:
    # From previous frame
    y_ori = worms_t[worm][counter-1][1]
    x_ori = worms_t[worm][counter-1][2]
    img_wormbody_old = worms_t[worm][counter-1][3]
    
    # Load new image and select roi based on previous frame
    img_new = cv.imread(img_dirs+'/'+img_name, -1)
    img_new_out = np.copy(img_new)
    img_new_roi = img_new[y_ori : y_ori + window, x_ori : x_ori + window]

    #TODO: try to use fastNlMeansDenoising to denoise
    # Find out wormbody based on some rules
    #img_new_roi_th = (cv.GaussianBlur(img_new_roi, (31,31), 0)
    #                  -cv.GaussianBlur(img_new_roi, (3,3), 0))
    img_wormbody = extendBody(img_new_roi, img_wormbody_old, counter, worm)

    # Find new bounday, recenter wormbody
    pos_wormbody = np.where( img_wormbody > 0)
    l = window//2
    if (len(pos_wormbody[0]) > 1): # to avoid unacceptable result
        y_mid = (min(pos_wormbody[0] + max(pos_wormbody[0])))//2 + y_ori
        x_mid = (min(pos_wormbody[1] + max(pos_wormbody[1])))//2 + x_ori
        y_ori_new =  y_mid - l 
        x_ori_new =  x_mid - l 
        pos_wormbody = list(zip(pos_wormbody[0] + y_ori, pos_wormbody[1] + x_ori))
        mask_edge = edgeScanner(img_wormbody)

        p1 = (x_ori, y_ori)
        p2 = (x_ori + window, y_ori + window)

        img_wormbody_mask = np.zeros(img_wormbody.shape, dtype=np.uint16)
        for i in pos_wormbody:
            img_wormbody_mask[i[0]-y_ori_new, i[1]-x_ori_new] = 1 
        img_wormbody = (img_new[y_ori_new : y_mid+l+1, x_ori_new : x_mid+l+1]
                        * img_wormbody_mask)

        img_wormbody = img_wormbody*evalBody(img_wormbody,worm,counter,
                img_new, x_ori_new, y_ori_new)
        # draw black egde in orignal image
        img_new_out[y_ori : y_ori + window, x_ori : x_ori + window] = \
            img_new_out[y_ori:y_ori+window, x_ori:x_ori+window] * (1-mask_edge)
        cv.imwrite(out_dirs_sw + '/' + img_name , img_wormbody)
        cv.rectangle(img_new_out, p1, p2, (65536, 65536, 65536), 1, cv.LINE_4)
    else :
        cv.putText(img_new, "Tracking fail", (100,80,), cv.FONT_HERSHEY_SIMPLEX,
            0.75,(0,0,255),2)
        print("Track Fail")
    worm_t[counter] = (counter, y_ori_new, x_ori_new, img_wormbody)
    worms_t[worm] = worm_t
    rows = [worm, counter/fps, y_ori_new, x_ori_new, img_name]
    spamwriter.writerow(rows)
    cv.imwrite(out_dirs_img + '/'+ img_name , img_new_out)

    counter = counter  + 1
    print("Frame: ", counter)

csvfile.close()
