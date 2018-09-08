#!/bin/python3
'''Script to represent worm track and statistics
|Version | Author | Commit
|0.1     | ZhouXY | first runable version
|0.2.0     | H.F.   | don't show the image but only save image
'''

import os,sys
import re
import numpy as np
from numpy import genfromtxt
import linecache
import matplotlib.pyplot as plt

# Plot the track image from([x], [y], frame rate)
# To Do: add distribution aside image
def PointPlot(pos_data, path, fram_rate):
    plt.clf()
    print(pos_data.shape)
    fig,ax = plt.subplots()
    ax.set_xlim([0,2000])
    ax.set_ylim([0,2000])
    print("Frame",len(pos_data))
    for i in range(len(pos_data)):
        x = pos_dat[i,0::2]
        y = pos_dat[i,1::2]
        ax.plot(x,y,'ro',MarkerSize = 0.5)
        ax.set_title("Frame: %i  %is/%is"%(i, i/fram_rate, len(pos_data)/fram_rate))
        plt.savefig("%s/%i.png"%(path,i),dpi=300) # save the image to folder

#
def ExtractValidPoint(pos_dat):
    dis = 0
    valid_points = pos_dat[:,(0,1)] # init point matrix with temp data
    for i in np.arange(pos_dat.shape[1]//2):
        [x,y] = (pos_dat[:, 2*i], pos_dat[:, 2*i+1])
        dis = 0
        for j in range(len(x)-1):
            distance = ((x[j+1]-x[j])**2+(y[j+1]-y[j])**2)**(1/2)
            if distance <40:
                dis+=distance
        if dis > 800:
            valid_points = np.column_stack((valid_points, x, y))
    return valid_points[1::] # delete the first temp row

def ExtractContinuesPoint(ValidPoints,gap = 5):
    dis = True
    count = 0
    ContinuesPoints = {}
    for i in ValidPoints:
        [x,y] = ValidPoints[i]
        for j in range(len(x)-1):
            distance = ((float(x[j+1])-float(x[j]))**2+(float(y[j+1])-float(y[j]))**2)**(1/2)
            if distance >140:
                count+=1
                if count == gap:
                    dis = False
                    break
        if dis:
            ContinuesPoints[i] = ValidPoints[i]
    return ContinuesPoints

if __name__ == "__main__":
    result_path = "/home/hf/iGEM/Results/20180904"
    pos_dat = genfromtxt('/home/hf/iGEM/Results/20180904/DiffussionTest7min.local',delimiter='\t')
    PointPlot(pos_dat[:,(0,1)], "%s/PointImage"%result_path, 2)

    #print("points",len(Pointdynamics))
    #ValidPoints = ExtractValidPoint(Pointdynamics)
    #ValidPoints = ExtractValidPoint(pos_dat)
    #print("valid",(ValidPoints.shape[1]))

    #ContinuesPoints = ExtractContinuesPoint(ValidPoints)
    #print("Continues",len(ContinuesPoints))
    #PointPlot(LocationDenminationChange(ContinuesPoints))
    #PointSubPlot(LocationDenminationChange(ValidPoints))
