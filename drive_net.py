# -*- coding: utf-8 -*-
"""
By Daniel Ellis and Michael Hall
"""
import time
import serial
from picamera import PiCamera
import numpy as np
import os
import networks

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
MODEL_VERSION_2 = 0
MODEL_VERSION_3 = 1

MODEL_NAME = 'net_v{}_{}_{}.model'.format(MODEL_VERSION_1, MODEL_VERSION_2, MODEL_VERSION_3)
print(MODEL_NAME)

WIDTH = 160
HEIGHT = 90
LR = 1e-2

print("Loading model...")
t = time.time()
model, description = networks.net_v0_0_1(WIDTH, HEIGHT, LR, 0)
print("created graph" + description)
graph_time = time.time() - t
t = time.time()
model.load(os.path.join(os.curdir, "Models", MODEL_NAME))
load_time = time.time() - t
print("Loaded model!")

print("graph time: " + str(graph_time))
print("load time: " + str(load_time))

time.sleep(5)

ready = True
output = Output()
camera.start_recording(output, 'yuv')

while True:
    #print("-----------")
    t = time.time()
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
        print('loop time: '+ str(loop_time))
    else:
        pass



camera.stop_recording()
