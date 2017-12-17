# -*- coding: utf-8 -*-
"""
compile_img.py
"""

import numpy as np
import cv2
import os

#check files in directory
print(os.listdir())

moves = np.genfromtxt(os.path.join(os.curdir,'moves.csv'), delimiter=',') #list of moves associated with pictures

filelist = os.listdir(os.path.join(os.curdir,'img')) #list of all the filenames of the images--includes .npy extension
lengthdir = len(filelist) #number of images

#print to see if moves are in correct format
print(moves)

train_data = [] #final list to feed to the neural net -- entries of format [image, [left, forward, right]]

for i in range(lengthdir):
    
    file = filelist[i] #iterate throught files
    

    img = np.load(os.path.join(os.curdir,'img', file)) # load image using np.load becasue files are .npy
    print(np.shape(img))
    #img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) 

    
    print(moves[i])
    curr_move = moves[i] #set moves for image
    
    if curr_move == 1: 
        curr_moves_array = np.array([0,1,0])
        image_move_data = [img, curr_moves_array] #format data    
        train_data.append(image_move_data)
    #train_data.append([img, [0,0,0]])
    elif curr_move == 3:
        curr_moves_array = np.array([1,0,0])
        image_move_data = [img, curr_moves_array] #format data    
        train_data.append(image_move_data)
    elif curr_move == 2:
        curr_moves_array = np.array([0,0,1])
        image_move_data = [img, curr_moves_array] #format data    
        train_data.append(image_move_data)
        
    print(file, curr_moves_array)

    if len(train_data) % 500 == 1: #save every 500 images and at end
        print(len(train_data))
        np.save('trainingdata.npy', train_data)

#remove bad data
#for i in range(849, 858):
#    del train_data[i]
#    print('deleting'+ str(i))

np.save('trainingdata.npy', train_data)
    
data = np.load('trainingdata.npy')
newimg = data[34][0] #check data validity by showing the 68th image in the file
otherimg = data[25][0]

while True:
    cv2.imshow("thing", newimg)
    cv2.imshow("other", otherimg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
