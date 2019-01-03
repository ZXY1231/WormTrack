#/bin/python3

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import sys
import argparse



#cv.imwrite('output.png', img)

if __name__ == '__main__':
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
    pos.append((876, 759, 887, 772))
    pos.append((892, 763, 910, 776))
    '''
    while True:
        line = input()
        if line:
            pos
    '''
    bg = (889, 679, 911,701)
    
    #fluo = np.zeros([len(files), len(pos)], dtype=np.float)
    fluo_0 = np.zeros([len(files), len(pos)]) # raw fluorescnece
    fluo_a = np.zeros([len(files), len(pos)]) # autofluorescence

    frame = 0
    fluo_t = np.zeros(len(pos))
    fluo_a_t = np.zeros(len(pos))
    for img_name in files:
        img = cv.imread(img_dirs+'/'+img_name, 2)
        img_bg = img[bg[0]:bg[2], bg[1]:bg[3]]
        for neuron in range(len(pos)):
            v_0 = pos[neuron][0]
            h_0 = pos[neuron][1]
            v_1 = pos[neuron][2]
            h_1 = pos[neuron][3]
            cv.imwrite('fluo/'+img_name+'%d'%(neuron)+'.tif',img[v_0:v_1, h_0:h_1])
            fluo_t[neuron] = np.mean(img[v_0:v_1, h_0:h_1])
            fluo_a_t = np.mean(img_bg)
        fluo_0[frame] = fluo_t
        fluo_a[frame] = fluo_a_t
        frame = frame + 1
    ## remove auto fluorescence
    fluo_ex = fluo_0 - fluo_a
    fps = 1
    t = np.linspace(0, len(fluo_0)-1, len(fluo_0)) * fps
    fluo_t0 = []
    for neuron in range(len(pos)):
        fluo_t0.append(np.mean(fluo_ex[0:5, neuron]))
    fluo_normal = fluo_ex/fluo_t0 - 1

    plt.figure(2, dpi=300)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    for neuron in range(len(pos)):
        plt.plot(t, fluo_normal[:, neuron],
                label='Neuron%d'%neuron + r', $I_0=$' + ' %d'%fluo_t0[neuron]) 
    plt.title('RGECO Fluorescence of Neuron AWA')
    plt.xlabel(r'\textbf{Time} (s)')
    plt.ylabel(r'\textit{$\Delta I_t//I_0$}')
    plt.legend()
    plt.savefig(out_dirs + '/'+ os.path.basename(img_dirs) +'.png')
    plt.show()
