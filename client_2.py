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
time.sleep(2)

#serial setup
ser = serial.Serial('/dev/ttyACM0', 9600)

#server setup
print("conecting to server")
PORT = 1337
HOSTNAME = '192.168.1.101'
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOSTNAME, PORT))
time.sleep(1)

x = 0
while True:
    #ts = time.time()
    data = s.recv(1)
    #print('Receiving Data:', time.time()-ts)
    if not data: 
        break
    if data == 7:
        ser.write(b'0')
        ser.close()
        s.close()
        print('The server has been closed!')
        break
    else:
        if data == 1:
            camera.capture(rawCamera, format="bgr")
            image = rawCamera.array
            np.save(str(x) + ".npy", image)
            ser.write(b'1')
            x += 1
        elif data == 2:
            camera.capture(rawCamera, format="bgr")
            image = rawCamera.array
            np.save(str(x) + ".npy", image)
            ser.write(b'2')
            x += 1
        elif data == 3:
            camera.capture(rawCamera, format="bgr")
            image = rawCamera.array
            np.save(str(x) + ".npy", image)
            ser.write(b'3')
            x += 1     
        elif data == 4:
            camera.capture(rawCamera, format="bgr")
            image = rawCamera.array
            np.save(str(x) + ".npy", image)
            ser.write(b'4')
            x += 1        
        elif data == 5:
            camera.capture(rawCamera, format="bgr")
            image = rawCamera.array
            np.save(str(x) + ".npy", image)
            ser.write(b'5')
            x += 1
        elif data == 6:
            camera.capture(rawCamera, format="bgr")
            image = rawCamera.array
            np.save(str(x) + ".npy", image)
            ser.write(b'6')
            x += 1   
        elif data == 0:
            ser.write(b'0')

        rawCamera.truncate(0)
        print(x, data)
        
        
        