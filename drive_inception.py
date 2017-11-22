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

camera = PiCamera(sensor_mode=4, resolution='160x90')
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

while True:
    print("-----------")
    t = time.time()
    try:
        camera.capture(y_data, 'yuv')
    except IOError:
        pass
    image = y_data[:90,:160]
    image_time = time.time() -t
    t = time.time()
    image = np.array(image).reshape(-1, WIDTH, HEIGHT, 1)
    reshape_time = time.time() - t 
    t = time.time()
    model_output = model.predict(image)
    predict_time = time.time() - t
    
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
    time.sleep(.2)
    ser.write(b'0')

    print('image time: ' + str(image_time) + '  ||  reshape time: '+ str(reshape_time) + "  ||  predict time: " + str(predict_time))
