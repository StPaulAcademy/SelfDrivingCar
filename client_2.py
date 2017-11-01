# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:20:09 2017

Daniel Ellis and Michael Hall

For use on the raspberry Pi
"""

import socket
import time
import serial
import picamera
from picamera.array import PiRGBArray
import numpy as np

#Initialize camera
camera = picamera.PiCamera()
rawCamera= PiRGBArray(camera)
camera.resolution = (100, 100)
camera.color_effects = (128,128)
time.sleep(2)

#serial setup
ser = serial.Serial('/dev/ttyACM0', 9600)

#server setup
print("conecting to server")
PORT = 1337
HOSTNAME = '192.168.1.100'
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOSTNAME, PORT))
time.sleep(1)
print("connected!")
x = 0
s.send(b'9')
while True:
    ts = time.time()
    data = s.recv(1)
    print('recieve data' + str(time.time()-ts))
    ts = time.time()
    if not data: 
        break
    if data == 7:
        ser.write(b'0')
        ser.close()
        s.close()
        print('The server has been closed!')
        break
    else:
        ser.write(data)
        if data == b'1' or data == b'2' or data == b'3':
            print('serial write' + str(time.time()-ts))
            ts = time.time()
            camera.capture(rawCamera, format="bgr")
            image = rawCamera.array
            print(np.shape(image))
            np.save(str(x) + ".npy", image)
            print('camera write' + str(time.time()-ts))
            ts = time.time()
            x += 1
            s.send(b'9')
#        elif data == 2:
#            camera.capture(rawCamera, format="bgr")
#            image = rawCamera.array
#            np.save(str(x) + ".npy", image)
#            ser.write(b'2')
#            x += 1
#        elif data == 3:
#            camera.capture(rawCamera, format="bgr")
#            image = rawCamera.array
#            np.save(str(x) + ".npy", image)
#            ser.write(b'3')
#            x += 1     
#        elif data == 4:
#            camera.capture(rawCamera, format="bgr")
#            image = rawCamera.array
#            np.save(str(x) + ".npy", image)
#            ser.write(b'4')
#            x += 1        
#        elif data == 5:
#            camera.capture(rawCamera, format="bgr")
#            image = rawCamera.array
#            np.save(str(x) + ".npy", image)
#            ser.write(b'5')
#            x += 1
#        elif data == 6:
#            camera.capture(rawCamera, format="bgr")
#            image = rawCamera.array
#            np.save(str(x) + ".npy", image)
#            ser.write(b'6')
#            x += 1   
#        elif data == 0:
#            ser.write(b'0')

        rawCamera.truncate(0)
        print(x, data)
        
        
        
