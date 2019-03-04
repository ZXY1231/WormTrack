#!python3
'''
Generate roi info of worm
| Version | Commit
| 0.1     | hf 

Usage: python3 trackingroi.py input_dir output_dir

key  : function
--------------------------------------
o    : diable/enable zoom
enter: select current rectangle as roi
mouse: draw, resize, move rectangle
'''
import os
import sys
import argparse
import cv2 as cv
import time
from matplotlib import pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as mpatch
import numpy as np


## Pass through all parameters
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()
img_dirs = args.path
out_dirs = args.output
log_time = time.strftime("%m%d%H%M", time.localtime())

white = (65536, 65536, 65536)

files = os.listdir(img_dirs)
files.sort()

rois_file = open(out_dirs+'/'+files[-1]+'-roi.csv', "w+")
rois = {}  # {label : x1, y1, w, h}
label = 0  # current roi label
fig, current_ax = plt.subplots()
x1, y1 = 0, 0 # current roi location
x2, y2 = 0, 0
img = cv.imread(img_dirs+'/'+files[-1], -1)
img_out = np.copy(img)

def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    global x1, y1, x2, y2
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

def toggle_selector(event):
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)
    if event.key in ['enter']:
        global label
        w = abs(x2-x1)
        h = abs(y2-y1)
        roi = mpatch.Rectangle((x1,y1), w, h, fill=False)
        current_ax.add_patch(roi)
        rois[label] = (x1, y1, w, h)
        current_ax.annotate(label, (x1,y1), fontsize=6)
        cv.rectangle(img_out, (x1, y1), (x2, y2), (65536, 65536, 65536), 1)
        cv.putText(img_out, '%d'%label, (x1, y1), cv.FONT_HERSHEY_SIMPLEX,0.7,
                white, 1, cv.LINE_AA)
        rois_file.write("%d\t%s\t%s\t%s\t%s\n"%(label, x1, y1, w, h))
        label += 1 # update label
        print('Select worm:', label)
    if event.key in ['d', 'D']:
        print('d')

plt.imshow(img)
toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                       #useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       #minspanx=5, minspany=5,
                                       interactive=True)
plt.connect('key_press_event', toggle_selector)
plt.show()
cv.imwrite(out_dirs+'/'+files[-1]+'-roi.tif', img_out)
rois_file.close()
