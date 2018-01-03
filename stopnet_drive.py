#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 2 2018

By Michael Hall and Daniel Ellis

Uses net_v0_0_1 and stopnet_v0_5
"""


import time
import serial
from picamera import PiCamera
import numpy as np
import os
import networks
import stopnet_networks
import tensorflow as tf

#Setup camera image and class for output
image = np.zeros((90,160)) 
WIDTH = 160
HEIGHT = 90
LR = 1e-2

class Output(object): 
    global image
    def write(self, buf):
        global ready
        global image
        y_data = np.frombuffer(
            buf, dtype=np.uint8, count=160*96).reshape((96, 160))
        if ready == True:
            image = y_data[:90, :160]
            ready = False

    def flush(self):
        pass

def init_drivenet(v1, v2, v3): #function for loading driving model parameters are version numbers
    global drivenet
    MODEL_NAME = 'net_v{}_{}_{}.model'.format(v1, v2, v3)
    print(MODEL_NAME)
    print("Loading model graph...")
    with tf.Graph().as_default(): #to reset to default graph
        t = time.time()
        drivenet, description = networks.net_v0_0_1(WIDTH, HEIGHT, LR, 0)
        print("created graph" + description)
        graph_time = time.time() - t
        t = time.time()
        drivenet.load(os.path.join(os.curdir, "Models", MODEL_NAME))
        load_time = time.time() - t
        print("Loaded model!")
        
        print("graph time: " + str(graph_time))
        print("load time: " + str(load_time))
    
def init_stopnet(v1, v2): #loads stopnet, parameters are version numbers
    global stopnet
    STOPNET_MODEL = 'stopnet_v{}_{}.model'.format(v1, v2)
    print("Loading stopnet...")
    with tf.Graph().as_default(): #to reset to default graph
        t = time.time()
        stopnet, description = stopnet_networks.stopnet(WIDTH, HEIGHT, LR, 0, 3, 3, 3)
        print("created graph" + description)
        graph_time = time.time() - t
        t = time.time()
        stopnet.load(os.path.join(os.curdir, "Models", STOPNET_MODEL))
        load_time = time.time() - t
        print("Loaded model!")
        
        print("graph time: " + str(graph_time))
        print("load time: " + str(load_time))


#call functions to initialize networks
init_drivenet(0,0,1)
init_stopnet(0,5)

#create camera object
camera = PiCamera(sensor_mode=4, resolution='160x90', framerate=40)
y_data = np.empty((96,160), dtype=np.uint8)
time.sleep(2)

#start serial connection
ser = serial.Serial('/dev/ttyACM0', 9600)
print("Opened serial connection!")
time.sleep(2)

#start camera recording
ready = True
output = Output()
camera.start_recording(output, 'yuv')

while True:
    #print("-----------")
    t = time.time()
    if ready == False: #if there is a new image available
        image = np.array(image).reshape(-1, WIDTH, HEIGHT, 1) #reshape for net
        stopnet_prediction = stopnet.predict(image)
        stopnet_prediction = np.argmax(stopnet_prediction[0])
        
        if stopnet_prediction == 0: #if there is a stopsign then stop
            print('found a stopsign!!')
            ser.write(b'0')
            break
        else: #if not then drive
            
            drivenet_output = drivenet.predict(image)
            drivenet_prediction = np.argmax(drivenet_output[0])
        
            if drivenet_prediction == 0:
                ser.write(b'3')
                print('prediction: ' + str(drivenet_output) + '  ||  action: left')
            if drivenet_prediction == 1:
                ser.write(b'1')
                print('prediction: ' + str(drivenet_output) + '  ||  action: forward')
            if drivenet_prediction == 2:
                ser.write(b'2')
                print('prediction: ' + str(drivenet_output) + '  ||  action: right')
            ready = True
            loop_time = time.time()-t
           # print('loop time: '+ str(loop_time))
    else: #if there is no frame ready then just pass the loop
        pass

camera.stop_recording()
