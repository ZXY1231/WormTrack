#/bin/python3

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import sys
import argparse


# Pass through all parameters
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('output', type=str)
#parser.add_argument('name', type=str)
args = parser.parse_args()
img_dirs = args.path
out_dirs = args.output
#name = args.name

files = os.listdir(img_dirs)
files.sort()

fps = 0.6
sti_begin = 11
window_size = 3

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Local the neurons
pos = []
pos.append((1208,739,1230,754))
#pos.append((876, 759, 887, 772))
#pos.append((892, 763, 910, 776))
bg = (1208, 784, 1235, 803)
    
#fluo = np.zeros([len(files), len(pos)], dtype=np.float)
fluo_0 = np.zeros([len(files), len(pos)]) # raw fluorescnece
fluo_a = np.zeros([len(files), len(pos)]) # autofluorescence

# extra fluorescence
frame = 0
fluo_t = np.zeros(len(pos))
fluo_a_t = np.zeros(len(pos))
for img_name in files:
    img = cv.imread(img_dirs+'/'+img_name, 2)
    img = cv.GaussianBlur(img,(5,5), 4)

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

## remove auto fluorescence
fluo_ex = fluo_0 - fluo_a
t = np.linspace(0, len(fluo_0)-1, len(fluo_0)) * fps
fluo_t0 = []
for neuron in range(len(pos)):
    fluo_t0.append(np.mean(fluo_ex[0:5, neuron]))
fluo_normal = fluo_ex/fluo_t0 - 1

plt.figure(2)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
for neuron in range(len(pos)):
    fluo_normal_smooth = moving_average(fluo_normal[:, neuron], window_size)
    plt.plot(t[window_size-1:], fluo_normal_smooth, color='r',
            label='Neuron%d'%neuron + r', $I_0=$' + ' %d'%fluo_t0[neuron]) 
plt.title('RGECO Fluorescence of Neuron AWA')
plt.xlabel(r'\textbf{Time} (s)')
plt.ylabel(r'\textit{$\Delta I_t//I_0$}')
plt.ylim(-1.5, 3)
plt.xlim(0, 120)
# stimulate bar 
for stimulate in range(3):
    sti0 = stimulate*35 + sti_begin
    plt.fill_between(np.linspace(sti0, sti0+5), 3,-1.5, color='k', alpha=0.2)
    plt.text(sti0-4, 2.5, 'Diacetyl')
plt.legend()
plt.savefig(out_dirs + '/'+ os.path.basename(img_dirs) +'.png', dpi=300)
np.save(out_dirs + '/' + os.path.basename(img_dirs) +
        '_fluo_normal.npy', fluo_normal)

#plt.savefig(out_dirs + '/'+ name +'.png', dpi=300)
#plt.show()
