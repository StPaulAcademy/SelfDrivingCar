# -*- coding: utf-8 -*-
#Created by Daniel Ellis and Michael Hall

import numpy as np
import random
import cv2
from scipy import ndimage

xcells = 2 #the number of cells to divide the image along the X axis
ycells = 2 #the number of cells to divide the image along the Y axis

sign = cv2.imread('stopsign.png') #THIS SHOULD BE THE SIGN
background = cv2.imread('background.jpg') #THIS SHOULD BE THE BACKGROUND

h1, w1 = background.shape[:2]
r = random.randint(-25, 25)

#newsign = ndimage.rotate(sign, angle = -r)
h2, w2 = sign.shape[:2]
scalefactor = int(min(h1, w1)/max(h2, w2)*100)
s = random.randint(scalefactor-50, scalefactor-10)/100
newsign = cv2.resize(sign, (0,0), fx=s, fy=s)
h3, w3 = newsign.shape[:2] #h3 and w3 are the final sizes used for the bounding box
newsign = ndimage.rotate(newsign, angle = -r)

h4, w4 = newsign.shape[:2]

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
print('Scaled by a factor of ' + str(s))
print('\n')
print('Global X coordinate of center: ' + str(xcoord))
print('Global Y coordinate of center: ' + str(ycoord))
print('\n')
print('Quadrant: ' + str(quadrant))
print('\n')
print('X coordinate at center: ' + str(xcenter))
print('Y coordinate at center: ' + str(ycenter))
print('\n')
print('Width from center : ' + str(w3/2))
print('Height from center : ' + str(h3/2))

'''UNCOMMENT THIS TO SEE A BOUNDING BOX'''
#cv2.rectangle(background, (int(xcoord-w3/2), int(ycoord-h3/2)), (int(xcoord+w3/2), int(ycoord+h3/2)), (255,0,0), 2)

cv2.imshow('', background)
cv2.waitKey(0)
cv2.destroyAllWindows()
