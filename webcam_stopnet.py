# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:59:14 2017

@author: 18MichaelRH
"""

import cv2
import numpy as np
import time
import stopnet_networks

VERSION_1 = 0
VERSION_2 = 4


print('loading model')
MODEL_NAME = 'stopnet_v{}_{}.model'.format(VERSION_1, VERSION_2)
model, description = stopnet_networks.stopnet(160, 90, 1e-3, 0, 3, 3, 3)

model.load(MODEL_NAME)
print('loaded!')

x = 0
#%%
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    
    img = cv2.resize(frame, (160, 90))
    
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img1.reshape(-1, 160, 90, 1)
    
    prediction = model.predict(img)
    
    output = np.argmax(prediction[0])
    if output == 0:
        cv2.rectangle(frame, (0,0), (250,25), (0,255,0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "STOP SIGN: " + str(prediction[0][0]), (0,20), font, .7, (255,255,255), 2, cv2.LINE_AA)
        print('yes')
        np.save(str(x)+'webcam_3.npy', img1)
        x += 1
        
    cv2.imshow('cap', frame)
    cv2.imshow('converted', img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #time.sleep(1)

cap.release()
cv2.destroyAllWindows()
