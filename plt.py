#!/bin/python3
import os
import argparse
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('img_file', type=str)
args = parser.parse_args()
img = args.img_file

img = cv.imread(img)
plt.imshow(img)
plt.show()




