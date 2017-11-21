# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:11:29 2017

By Daniel Ellis and Michael Hall
"""

from inceptionv3 import inceptionv3
import time
import serial
import picamera
from picamera.array import PiRGBArray
import numpy as np
camera = picamera.PiCamera()
rawCamera= PiRGBArray(camera)
camera.resolution = (160, 90)
camera.color_effects = (128,128)
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
    camera.capture(rawCamera, format='bgr')
    print("Captured image!")
    image = np.asarray(rawCamera.array)[:,:,0]
    image = np.array(image).reshape(-1, WIDTH, HEIGHT, 1)
    print("Formatted image!")
    print(np.shape(image))
    model_output = model.predict(image)
    print("Made prediction!")
    print(model_output)
    model_prediction = np.argmax(model_output[0])
    print("Got prediction value!")
    print(model_prediction)
    if model_prediction == 0:
        ser.write(b'3')
        print("Sent left command!")
    if model_prediction == 1:
        ser.write(b'1')
        print("Sent forward command!")
    if model_prediction == 2:
        ser.write(b'2')
        print("Sent right command!")
    time.sleep(.2)
    ser.write(b'0')
    rawCamera.truncate(0)
