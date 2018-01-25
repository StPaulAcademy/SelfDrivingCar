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
import multiprocessing as mp

#Initialize camera
camera = picamera.PiCamera()
rawCamera= PiRGBArray(camera)
camera.resolution = (160, 90)
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


save_ready = True
def save_img(data):
    global x, save_ready
    save_ready = False
    camera.capture(rawCamera, format="bgr")
    image = rawCamera.array
    if data == b'1':
        image = np.array([image[:,:,0], np.array([0,1,0])])
    if data == b'3':
        image = np.array([image[:,:,0], np.array([1,0,0])])
    if data == b'2':
        image = np.array([image[:,:,0], np.array([0,0,1])])
    np.save(str(x) + ".npy", image)
    x += 1
    rawCamera.truncate(0)

def reset_img():
    global save_ready
    save_ready = True

if __name__ == '__main__': 
    pool = mp.Pool()
    while True:
        ts = time.time()
        data = s.recv(1)
    
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
                if save_ready == True:
                    pool.apply_async(save_img, args = (data, ), callback = reset_img)
            print(x, data)
            s.send(b'9')
        
        
        
