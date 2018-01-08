# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:59:14 2017

@author: 18MichaelRH
"""

import cv2
import numpy as np
import time
import stopnet_networks
import os
from multiprocessing.dummy import Pool

VERSION_1 = 0
VERSION_2 = 5


print('loading model')
MODEL_NAME = 'stopnet_v{}_{}.model'.format(VERSION_1, VERSION_2)
model, description = stopnet_networks.stopnet(160, 90, 1e-3, 0, 3, 3, 3)

model.load(MODEL_NAME)
print('loaded!')

def search(x, y, w, h):
    img = img1[y:int(y + h), x:int(x + w)] 
    print(img.shape)
    img = cv2.resize(img, (160,90))
    img = img.reshape(-1, 160, 90, 1)
    prediction = model.predict(img)
    print(prediction)
    output = np.argmax(prediction[0])
    if output == 0:
        cv2.rectangle(img1, (x,y), (x+80, y+45), (0,255,0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "STOP SIGN: " + str(prediction[0][0]), (0,20), font, .7, (255,255,255), 2, cv2.LINE_AA)
        print('yes')


#%%
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        
        img = cv2.resize(frame, (160, 90))
        
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img1.shape
        pool = Pool()
        for dim in [[0, 0], [width/4, 0], [width/2, 0], [0, height/4], [0, height/2], [width/2, height/2]]:
            pool.apply_async(search, args=(dim[0],dim[1],int(width/2),int(height/2),))
        pool.close()
        pool.join()
        print("---------------")

            
        cv2.imshow('cap', frame)
        cv2.imshow('converted', img1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #time.sleep(1)
    
    cap.release()
    cv2.destroyAllWindows()
