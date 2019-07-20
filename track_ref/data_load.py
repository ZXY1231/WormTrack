import numpy as np
import os
import cv2 as cv
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
args = parser.parse_args()
img_dirs = args.path


files = os.listdir(img_dirs)
files.sort()


a = []

for img_name in files:
    img = cv.imread(img_dirs + '/' + img_name, -1)
    #a.append(img)



