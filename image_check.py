# -*- coding: utf-8 -*-
"""
image_check.py
Michael Hall and Daniel Ellis

Use to check if the images look good in a data set 
"""

import numpy as np
import cv2
import os

#check files in directory
print(os.listdir())

filelist = os.listdir(os.path.join(os.curdir,'img')) #list of all the filenames of the images--includes .npy extension
lengthdir = len(filelist) #number of images


for i in range(lengthdir):
    print(i)
    file = filelist[i] #iterate throught files
    

    img = np.load(os.path.join(os.curdir,'img', file)) # load image using np.load becasue files are .npy
    while True:
        cv2.imshow("thing", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): #q to iterate through the files to look
            break
    
cv2.destroyAllWindows()
