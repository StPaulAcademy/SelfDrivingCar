# -*- coding: utf-8 -*-
#Created by Daniel Ellis and Michael Hall

import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

xcells = 10 #the number of cells to divide the image along the X axis
ycells = 10 #the number of cells to divide the image along the Y axis

sign = cv2.imread('stopsign.png') #THIS SHOULD BE THE SIGN
background = cv2.imread('background.jpg') #THIS SHOULD BE THE BACKGROUND

h1, w1 = background.shape[:2]
r = random.randint(-25, 25)

newsign = ndimage.rotate(sign, angle = -r)
h2, w2 = newsign.shape[:2]

scalefactor = int(min(h1, w1)/max(h2, w2)*100)
s = random.randint(scalefactor-50, scalefactor-10)/100
newsign = cv2.resize(newsign, (0,0), fx=s, fy=s)

h3, w3 = newsign.shape[:2]

x = random.randint(0, w1-w3)
y = random.randint(0, h1-h3)

xcoord = x+w3/2
ycoord = y+h3/2

xcenter = (xcoord-((xcoord//(w1/xcells))*xcells))/xcells
ycenter = (ycoord-((ycoord//(h1/ycells))*ycells))/ycells

#Quadrants are enumerated individually along the X and Y axis starting at (0,0) in the top left corner and (n,n) in th ebottom right corner
xquadrant = xcoord//(w1/xcells)
yquadrant = ycoord//(h1/xcells)

background[y:y+h3, x:x+w3] = np.where(np.repeat(np.any(newsign > 0, axis=2), 3, axis=1).reshape(h3, w3, 3), newsign, background[y:y+h3, x:x+w3])

print('Rotated ' + str(r) + ' degrees about the center')
print('Scaled by a factor of ' + str(s))
print('\n')
print('Global X coordinate of center: ' + str(xcoord))
print('Global Y coordinate of center: ' + str(ycoord))
print('\n')
print('Quadrant on X axis: ' + str(xquadrant))
print('Quadrant on Y axis: ' + str(yquadrant))
print('\n')
print('X coordinate at center: ' + str(xcenter))
print('Y coordinate at center: ' + str(ycenter))
print('\n')
print('Width from center : ' + str(w2/2))
print('Height from center : ' + str(h2/2))

#plt.imshow(background)
cv2.imshow('', background)
cv2.waitKey(0)
cv2.destroyAllWindows()
