import os
import sys
import argparse
import multiprocessing as mp
import random
import string
import numpy as np
import cv2 as cv
import time
from tracklib.wormtracker import WorkTracker as wt

output = mp.Queue()

parser = argparse.ArgumentParser()
parser.add_argument('imgs', type=str)
parser.add_argument('output', type=str)
parser.add_argument('-a', '--auto', type=str, default='')
args = parser.parse_args()
img_dirs = args.imgs
out_dirs = args.output
is_auto = args.auto


fps = 2
#ori_imgs = {} # {img_key: img, img_name}dict to store all image
ori_imgs = mp.Manager().dict()
worm = {} # {worm_key: worm}
timestamp = time.strftime("%m%d%H%M", time.localtime())

# Load the image data into dictionary
def load_data(img_dirs):
    files = os.listdir(img_dirs)
    files.sort() # sort in dictonary order
    files.reverse()
    key = 0
    for img_name in files:
        ori_imgs[key]= (cv.imread(img_dirs+'/'+ img_name, -1), img_name)
        key += 1

# Initialize roi 
def tracker_create(is_auto, ori_imgs_l, out_dirs_l, timestamp_l):
    init_rois = {}
    if len(is_auto) == 0: # toggle roi selector
        print("by hand")
        #init_rois[0] = 
    else: # load roi files 
        # TODO: check roi file whether match
        with open(is_auto) as rois:
            for line in rois:
                data = [int(l) for l in line.split()]
                init_rois[data[0]] = data[1:]
    # Create new object
    if len(init_rois) > 0 : # at least exist one chosen worm
        for roi_key in init_rois.keys():
            x1, y1, x2, y2= init_rois.get(roi_key)
            out_dir = out_dirs_l + '/%d-%s'%(roi_key, timestamp)
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            #worm[roi_key] = wt(roi_key, x1, y1, x2, y2, ori_imgs_l, out_dir)
            worm[roi_key] = wt(roi_key, x1, y1, x2, y2, out_dir)

# 
def tracker_tracking(worm_key, output):
    worm[worm_key].tracker_init(ori_imgs[0][0])
    result = worm[worm_key].tracking(ori_imgs)
    output.put(result)


load_data(img_dirs)
tracker_create(is_auto, ori_imgs, out_dirs, timestamp)
# load in to process
processes = [mp.Process(target=tracker_tracking, args=(key, output)) 
        for key in worm.keys()]

for p in processes:
    p.start()

#for p in processes:
#    p.join()

results = [output.get() for p in processes]

print("Result")
#print(results)
