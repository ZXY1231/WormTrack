#!/bin/python3
'''Script to represent worm track and statistics
|Version | Author | Commit
|0.1     | ZhouXY | first runable version
|0.2.0   | H.F.   | don't show the image but only save image
|0.2.1   | H.F.   | use Absolute-value norm to choose continued worm
|0.2.2   | H.F.   | Draw the distrbution in two direction
ToDo: add quanlity report
'''

import numpy as np
import matplotlib.pyplot as plt

# Plot the track image from([x], [y], frame rate)
# To Do: add distribution aside image
def PointPlot(pos_data, path, fram_rate):
    plt.clf()
    for i in range(len(pos_data)): 
        plt.subplot(2,2,3)
        plt.xlim([0,2000])
        plt.ylim([0,2000])
        x = pos_data[i,0::2]
        y = pos_data[i,1::2]
        plt.plot(x,y,'ro',MarkerSize = 0.5)
        
        plt.subplot(2,2,1) # Histrogram in x-axis
        plt.hist(x, bins=20, color='r')
        plt.xlim([0,2000])
        plt.ylim([0,20])
        plt.title("Frame: %i  %is/%is"%(i, i/fram_rate, len(pos_data)/fram_rate))

        plt.subplot(2,2,4) # Histrogram in y-axis
        plt.hist(y, bins=20, color='r', orientation='horizontal')
        plt.ylim([0,2000])
        plt.xlim([0,20])

        print("drawing %i/%i frame"%(i, len(pos_data)))
        plt.savefig("%s/%i.png"%(path,i),dpi=200) # save the image to folder
        #plt.pause(0.002)

# Remove the freeze points, such as dust
def ExtractValidPoint(pos_data):
    dis = 0
    valid_points = pos_data[:,(0,1)] # init point matrix with temp data
    for i in np.arange(pos_data.shape[1]//2): # read each one worm localtion
        [x,y] = (pos_data[:, 2*i], pos_data[:, 2*i+1])
        if (max(x)-min(x)+max(y)-min(y) > 15):
            valid_points = np.column_stack((valid_points, x, y))
    return valid_points[:, 2::] # delete the first temp point

# Remove trait that are long discontinued segment
def ExtractContinuesPoint(pos_data,gap_num = 1):
    gap_len = 30
    dis = True
    count = 0
    continued_points = pos_data[:,(0,1)] # init point matrix with first worm
    pos_diff = pos_data[:-1:,:] - pos_data[1::,:] # postion diff between frame
    gap = (abs(pos_diff[:,0::2]) + abs(pos_diff[:,1::2]) > gap_len)
    continued = (sum(gap) < gap_num)
    for i in np.arange(continued.size): #  each single worm
        [x,y] = [pos_data[:, 2*i], pos_data[:,2*i+1]]
        if (continued[i]):
            continued_points = np.column_stack((continued_points,x,y))
    return continued_points[:, 2::] # delete 1st temp worm



if __name__ == "__main__":
    result_path = "/home/hf/iGEM/Results/20180904"
    pos_dat = np.loadtxt('%s/DiffussionTest7min.local'%result_path,delimiter='\t')
    #PointPlot(pos_dat[:,(0,1,2,3,4,5)], "%s/PointImage"%result_path, 2)
    ValidPoints = ExtractValidPoint(pos_dat)
    print("validpoints: %i"%(ValidPoints.shape[1]//2)) 
    #PointPlot(ValidPoints, "%s/ValidPoint"%result_path, 2)
    ContinuedPoints = ExtractContinuesPoint(ValidPoints)
    #np.savetxt("%s/continued.local"%result_path,ContinuedPoints,delimiter='\t')
    PointPlot(ContinuedPoints, "%s/ContinuedPoints"%result_path, 2)
