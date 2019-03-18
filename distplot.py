#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import os
import sys
import glob
import time
import argparse
'''
| Version | Commit
| 0.1     | for single pos_file from imagej 
| 0.2     | for many worm
'''
# Pass through all parameters
parser = argparse.ArgumentParser()
parser.add_argument('pos_file', type=str)
parser.add_argument('out_dir', type=str)
parser.add_argument('-l', '--list', type=str, default='')
args = parser.parse_args()
pos_file = args.pos_file
out_dir = args.out_dir
pos_list = args.list

timestamp = time.strftime("%m%d%H%M", time.localtime())
dist_out_dir = out_dir + 'dist'+timestamp
traj_out_dir = out_dir + 'traj'+timestamp
if not os.path.isdir(dist_out_dir):
    os.mkdir(dist_out_dir)
    os.mkdir(traj_out_dir)
# load experiment data
#files = os.listdir(img_dirs)
#frame0 = np.loadtxt(glob.glob(pos_file+'/0-*/*.csv')[0], delimiter='\t', skiprows=1, usecols=[1,2,3], dtype=np.int16)

pos_dat = None # numpy array [ worm[ time[ yx]]]
img_name = None
done = 0
if pos_list != '':
    data_list = []
    with open(pos_list, "r") as f:
        line = f.readline()
        while line:
            data_list.append(np.loadtxt(pos_file+line.strip(), delimiter='\t',
                skiprows=1, usecols=[2,3]))
            if done == 0:
                with open(pos_file+line.strip(), 'r') as name:
                    img_name = [line.strip().split('\t')[4] for line in name]
                done = 1
            line = f.readline()
    pos_dat = np.asarray(data_list, dtype=np.int16)

#pos_dat = np.loadtxt(pos_file,delimiter=',', skiprows=1)
scale = 12.5/3510
shift = 3510/2
fps = 2.5
x = (pos_dat[:,2,0]-shift)*scale
y = (pos_dat[:,2,1]-shift)*scale

# Fixing random state for reproducibility
#np.random.seed(19680801)
# the random data
#x = np.random.randn(1000)
#y = np.random.randn(1000)

nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8, 8))

#axScatter = plt.axes(rect_scatter)
#axHistx = plt.axes(rect_histx)
#axHisty = plt.axes(rect_histy)


w_total, t_total = pos_dat.shape[0:-1]
# now determine nice limits by hand:
lim = 6.5 # 6cm
#lim = (int(xymax/binwidth) + 1) * binwidth
binwidth = 0.5
for t_i in range(t_total-1, 0, -1):
    y = (pos_dat[:,t_i,0]-shift) * scale
    x = (pos_dat[:,t_i,1]-shift) * scale
    # the scatter plot:
    axScatter = plt.axes(rect_scatter)
    plt.cla()#
    axScatter.scatter(x, y, c ='b')
    
    #xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))

    
    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx = plt.axes(rect_histx)
    plt.cla()
    axHistx.hist(x, bins=bins, color = 'b')
    axHisty = plt.axes(rect_histy)
    plt.cla()
    axHisty.hist(y, bins=bins, orientation='horizontal', color = 'b')
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    axHistx.set_xlim(axScatter.get_xlim())
    axHistx.set_ylim((0, 40))
    axHisty.set_ylim(axScatter.get_ylim())
    axHisty.set_xlim((0, 40))
    plt.title(r'Distribution of trained 0810 Worm: %0.1fs'%((t_total-t_i)/fps),
            y=-0.1, x=-1.8)
    plt.savefig(dist_out_dir +'/'+img_name[t_i] + '.png')#, dpi=100)
    print(t_i)
