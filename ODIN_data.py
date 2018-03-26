#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:55:25 2018

@author: michael
"""

import numpy as np
import cv2
print('this actually runs')
file = open('info.lst')

filedata = file.readlines()

data = []
i = 0
for line in filedata:

    items = line.split()
    x = float(items[2])/160
    y = float(items[3])/160
    w = float(items[4])/160
    h = float(items[5])/160
    
    img = cv2.imread('./info/{}'.format(items[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if x < .5:
        if y < .5: #TOP LEFT
            data.append([img, [[1, x, y, w, h, 0],
                         [0, 0,0,0,0,0],
                         [0, 0,0,0,0,0],
                         [0, 0,0,0,0,0]]])
        else:#BOTTOM LEFT
            y -= .5
            data.append([img, [[0, 0,0,0,0,0],
                        [1, x, y, w, h, 0],
                        [0, 0,0,0,0,0],
                        [0, 0,0,0,0,0]]])
    else:
        x -= .5
        if y < .5: #TOP RIGHT
            data.append([img, [[0, 0,0,0,0,0],
                         [0, 0,0,0,0,0],
                         [1, x, y, w, h, 0],
                         [0, 0,0,0,0,0]]])
        else: #BOTTOM RIGHT
            y -= .5
            data.append([img, [[0, 0,0,0,0,0],
                        [0, 0,0,0,0,0],
                        [0, 0,0,0,0,0],
                        [1, x, y, w, h, 0]]])
    
    data.append([img, [[0, 0,0,0,0,0],
                       [0, 0,0,0,0,0],
                       [0, 0,0,0,0,0],
                       [0, 0,0,0,0,0]]])
    print(i, len(data))
    i += 1
np.save('ODINdata.npy', data)
file.close()