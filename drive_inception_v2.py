# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:11:29 2017

By Daniel Ellis and Michael Hall
"""

from inceptionv3 import inceptionv3
import time
import serial
from picamera import PiCamera
import numpy as np

image = np.zeros((90,160))

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

camera = PiCamera(sensor_mode=4, resolution='160x90', framerate=40)
y_data = np.empty((96,160), dtype=np.uint8)
time.sleep(2)

ser = serial.Serial('/dev/ttyACM0', 9600)
print("Opened serial connection!")

MODEL_VERSION_1 = 0
MODEL_VERSION_2 = 4

MODEL_NAME = 'APTnet-v{}-{}.model'.format(MODEL_VERSION_1, MODEL_VERSION_2)
print(MODEL_NAME)

WIDTH = 160
HEIGHT = 90
LR = 1e-2

print("Loading model...")
model = inceptionv3(WIDTH, HEIGHT, LR)
print("created graph")

model.load(MODEL_NAME)

print("Loaded model!")

ready = True
output = Output()
camera.start_recording(output, 'yuv')
t = time.time()
while True:
    print("-----------")

    if ready == False:
        image = np.array(image).reshape(-1, WIDTH, HEIGHT, 1)
        model_output = model.predict(image)
        model_prediction = np.argmax(model_output[0])
    
        if model_prediction == 0:
            ser.write(b'3')
            print('prediction: ' + str(model_output) + '  ||  action: left')
        if model_prediction == 1:
            ser.write(b'1')
            print('prediction: ' + str(model_output) + '  ||  action: forward')
        if model_prediction == 2:
            ser.write(b'2')
            print('prediction: ' + str(model_output) + '  ||  action: right')
        ready = True
        loop_time = time.time()-t
        t = time.time()
    else:
        #ser.write(b'0')
        #time.sleep(0.1)
        pass
    print('loop time: '+ str(loop_time))

camera.stop_recording()
