# -*- coding: utf-8 -*-
"""
drive_inception_v3.py

Drives ATLAS using video recording and async image transfer to the prediction
Driving is continuous and relatively smooth

This sends the motor angle to the arduino
function to use is 90 + 45*(L - R) = angle to set motor

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
            ser.write(b'90')
            print('prediction: ' + str(model_output) + '  ||  action: left')
        if model_prediction == 1:
            angle  = 90 + 45*(model_output[0][0]-model_output[0][2])
            ser.write(bytes(str(angle), 'utf-8'))
            print('prediction: ' + str(model_output) + '  ||  action: forward')
        if model_prediction == 2:
            angle  = 90 + 45*(model_output[0][0]-model_output[0][2])
            ser.write(bytes(str(angle), 'utf-8'))
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
