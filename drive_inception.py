# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:42:05 2017

@author: 18MichaelRH
"""

from inceptionv3 import inceptionv3
import time
import serial
import picamera
from picamera.array import PiRGBArray
import numpy as np

#Initialize camera
camera = picamera.PiCamera()
rawCamera= PiRGBArray(camera)
camera.resolution = (160, 90)
camera.color_effects = (128,128)
time.sleep(2)

#serial setup
ser = serial.Serial('/dev/ttyACM0', 9600)

while True:
    t = time.time()
    
    #take picture
    camera.cature(rawCamera, format='bgr')
    image = rawCamera.array