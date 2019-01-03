#/bin/python3
import os
import matplotlib.pyplot as plt
import cv2 as cv
import sys
import argparse

"""
Just track, don't classify

| Version | Comment
|   0.1   | Unfinish

"""


# Pass through all parameters
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()
img_dirs = args.path
out_dirs = args.output

files = os.listdir(img_dirs)
files.sort()

# Local the neurons
pos = []
#pos.append((876, 759, 887, 772))
pos.append((876, 759, 887, 772))
pos.append((892, 763, 910, 776))
bg = (889, 679, 911,701)

fluo_0 = np.zeros([len(files), len(pos)]) # raw fluorescnece
fluo_a = np.zeros([len(files), len(pos)]) # autofluorescence

# extra fluorescence
frame = 0
fluo_t = np.zeros(len(pos))
fluo_a_t = np.zeros(len(pos))
for img_name in files:
    img = cv.imread(img_dirs+'/'+img_name, 2)
    img_bg = img[bg[0]:bg[2], bg[1]:bg[3]]
    for neuron in range(len(pos)):
        h_0 = pos[neuron][0]
        v_0 = pos[neuron][1]
        h_1 = pos[neuron][2]
        v_1 = pos[neuron][3]
        fluo_t[neuron] = np.mean(img[v_0:v_1, h_0:h_1])
        fluo_a_t = np.mean(img_bg)
    fluo_0[frame] = fluo_t
    fluo_a[frame] = fluo_a_t
    #cv.rectangle(img, (v_0,h_0), (v_1, h_1))
    #jcv.imshow(img)
    #cv.show()
    #cv.imwrite(img, )
    frame = frame + 1
