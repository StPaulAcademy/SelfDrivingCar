# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:29:56 2017

@author: 18DanielBE
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

VERSION_ONE = 0
VERSION_TWO = 2
MODEL_NAME = "APTnet-v{}-{}.model".format(VERSION_ONE, VERSION_TWO)
print(MODEL_NAME)

WIDTH = 160
HEIGHT = 90
LR = 0.01

print("Loading model...")
model = inceptionv3(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)
print("Loaded model!")

while True:
    camera.capture(rawCamera, format='bgr')
    print("Captured image!")
    image = np.asarray(rawCamera.array)[:,:,0]
    image = np.asarray(image)
    print(np.shape(image))
    print("Formatted image!")
    model_output = model.predict(image)
    print("Made prediction!")
    print(model_output)
    model_output_array = np.asarray(model_output)
    print("Formatted prediction!")
    print(model_output_array)
    model_prediction = np.argmax(model_output_array)
    print("Got prediction value!")
    print(model_prediction)
    if model_prediction == 0:
        self.motor.setSpeed(self.speed)
        self.motor.run(Adafruit_MotorHAT.FORWARD)
        self.servo.ChangeDutyCycle(7.0)
        print("Sent left command!")
    if model_prediction == 1:
        self.motor.setSpeed(self.speed)    
        self.motor.run(Adafruit_MotorHAT.FORWARD)
        self.servo.ChangeDutyCycle(2.5)
        print("Sent forward command!")
    if model_prediction == 2:
        self.motor.setSpeed(self.speed)    
        self.motor.run(Adafruit_MotorHAT.FORWARD)
        self.servo.ChangeDutyCycle(12.5)
        print("Sent right command!")
    else:
        self.servo.ChangeDutyCycle(7.0)
    
    
