# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 08:18:27 2017
Authors: Michael Hall and Daniel Ellis
Advanced Technology Projects 2017

predict_alexnet.py
"""

from alexnet import alexnet
import cv2
import time
import picamera
from picamera.array import PiRGBArray
import numpy as np
import serial

ser = serial.Serial('/dev/ttyACM0', 9600)#start serial connection

MODEL_VERSION = 0.1 #model version to use--needs to be in dir
MODEL_NAME = "ATPnet-alexnet-v{}.model".format(MODEL_VERSION)

WIDTH = 150 #parameters for net
HEIGHT = 150
LR = 1e-3

print("loading alexnet with weights...")
model = alexnet(WIDTH, HEIGHT, LR) #load alexnet and weights
model.load(MODEL_NAME)
print("loaded!")

camera = picamera.PiCamera() #camera setup
rawCamera= PiRGBArray(camera)
time.sleep(2)

print("driving!")
while True:
    lasttime_breakdown = time.time()
    lasttime_loop = time.time()
    #capture image as array
    camera.capture(rawCamera, format="bgr")
    img = rawCamera.array    
    #scale and cvt color to grayscale
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) 
    img = img[120:600][:] # 1:1 aspect ratio
    img = cv2.resize(img, (150,150))
    
    print("camera took {} seconds".format(time.time()-lasttime_breakdown))
    lasttime_breakdown = time.time()
    
    model_output = list(np.around(model.predict(img)[0]))#what is 0th element of model.predict for?
    
    print("prediction took {} seconds".format(time.time()-lasttime_breakdown))
    lasttime_breakdown = time.time()  
    
    if model_output == [1,0,0]: #convert onehot to numerical and send to serial bus
        ser.write(b'3')
    if model_output == [0,1,0]:# potential problem is overloading the serial connection by running the network to much. 
        ser.write(b'1')
    if model_output == [0,0,1]:
        ser.write(b'2')
   
    print("serial took {} seconds".format(time.time()-lasttime_breakdown))
        
    print("whole loop took {} seconds".format(time.time()-lasttime_loop))          
    rawCamera.truncate(0) #clear camera