import os
import sys
import args
import multiprocessing as np
import random
import string
import numpy as np
import cv2 as cv
from wormtracker import WormTracker

output = mp.Queue()

parser = argparse.ArgumentParser()
parser.add_argument('imgs', type=str)
parser.add_argument('output', type=str)
parser.add_argument('-a', '--auto', type=int, default=1)
parser.add_argument('-a', '--auto', type=str, default=0)
args = parser.parse_args()
img_dirs = args.imgs
out_dirs = args.output
is_auto = args.auto


fps = 2
ori_imgs = {} # {img_key: img, img_name}dict to store all image
worm = {} # {worm_key: worm_t}

# Load the image data into dictionary
def load_data(img_dirs):
    files = os.listdir(img_dirs)
    files.sort() # sort in dictonary order
    files.reverse()
    key = 0
    for img_name in files:
        ori_imgs[img_name]= (img_name, cv.imread(img_dirs+'/'+ img_name, -1))
        key += 1

#
def tracker_create(model, roi_file, is_auto):
    init_rois = {}
    if is_auto == 0: # toggle roi selector
        print("by hand")
        #init_rois[0] = 
    else: # load roi files
        with open(is_auto) as rois:
            for line in rois:
                data = [int(l) for l in line.split('\t')]
                init_rois[data[0]] = data[1, -1]
        print(init_rois)

    if len(init_rois) > 0 : # at least exist one chosen worm
        for roi_key in init_rois.keys():
            print(init_rois.getitems(roi_key))
            x1, y1, x2, y2= init_rois.getitems(roi_key)
            worm[roi_key] = WormTracker(roi_key, x1, y1, x2, y2)
            # create new object

# 
def tracker_tracking(ori_imgs, worm_key, output):
    result = worm[worm_key].tracking(ori_imgs)
    output.put(result)

precesses = [mp.Process(target=tracker_tracking, args=(5, output)) for x in range(4)]

for p in processes:
    p.start()

for p in processes:
    p.join()

results = [output.get() for p in processes]

print(results)
