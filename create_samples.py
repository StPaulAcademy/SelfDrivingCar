# -*- coding: utf-8 -*-
#Created by Daniel Ellis and Michael Hall

import numpy as np
import random
import cv2
from scipy import ndimage

xcells = 2 #the number of cells to divide the image along the X axis
ycells = 2 #the number of cells to divide the image along the Y axis

sign = cv2.imread('Stop_sign.png') #THIS SHOULD BE THE SIGN
background = cv2.imread('car_1.jpg') #THIS SHOULD BE THE BACKGROUND
background = cv2.resize(background, (160,160))
h1, w1 = background.shape[:2]
r = random.randint(-25, 25)

#newsign = ndimage.rotate(sign, angle = -r)
h2, w2 = sign.shape[:2]
scalefactor = int(min(h1, w1)/max(h2, w2)*100)
scale = random.randint(scalefactor-60, scalefactor-40)/100
newsign = cv2.resize(sign, (0,0), fx=scale, fy=scale)
h3, w3 = newsign.shape[:2]
newsign = ndimage.rotate(newsign, reshape = True, angle = -r)

h4, w4 = newsign.shape[:2]
print(w1, h1, w4, h4)
x = random.randint(0, w1-w4)
y = random.randint(0, h1-h4)

xcoord = x+w4/2
ycoord = y+h4/2

xcenter = (xcoord-((xcoord//(w1/xcells))*xcells))/xcells
ycenter = (ycoord-((ycoord//(h1/ycells))*ycells))/ycells

#Quadrants are enumeraed from left to right starting at 0. The start of each new row picks up on where the last row ended
quadrant = (xcoord//(w1/xcells))+(ycells*(ycoord//(h1/xcells)))

background[y:y+h4, x:x+w4] = np.where(np.repeat(np.any(newsign > 0, axis=2), 3, axis=1).reshape(h4, w4, 3), newsign, background[y:y+h4, x:x+w4])

print('Rotated ' + str(r) + ' degrees about the center')
print('Scaled by a factor of ' + str(scale))
print('\n')
print('Global X coordinate of center: ' + str(xcoord))
print('Global Y coordinate of center: ' + str(ycoord))
print('\n')
print('Quadrant: ' + str(quadrant))
print('\n')
print('X coordinate at center: ' + str(xcenter))
print('Y coordinate at center: ' + str(ycenter))
print('\n')
print('Width : ' + str(w4))
print('Height : ' + str(h4))

'''UNCOMMENT THIS TO SEE A BOUNDING BOX'''
cv2.rectangle(background, (int(xcoord-w4/2), int(ycoord-h4/2)), (int(xcoord+w4/2), int(ycoord+h4/2)), (255,0,0), 2)

cv2.imshow('a', background)
cv2.waitKey(0)
cv2.destroyAllWindows()
