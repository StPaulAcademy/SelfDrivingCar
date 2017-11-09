import socket
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


HOST = ''
PORT = 1337
ser = serial.Serial('/dev/ttyACM0', 9600)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(5)
print('A server opened at host ' + socket.gethostname() + ' on port ' + str(PORT)  + '!')

conn, addr = s.accept()
print(addr[0] + ' has connected!')

x = 0

while True:
    
    data = conn.recv(1)
    
    if not data: 
        break
    
    if data == b'5':
        ser.write(data)
        camera.capture(rawCamera, format="bgr")
        image = rawCamera.array
        imagefinal = np.asarray(list(image[:,:,0] + [1, 0, 0]))
        np.save(str(x) + ".npy", image)
        rawCamera.truncate(0)
        conn.send(data)
        x += 1
        continue
        
    if data == b'4':
        ser.write(data)
        camera.capture(rawCamera, format="bgr")
        image = rawCamera.array
        imagefinal = np.asarray(list(image[:,:,0] + [0, 1, 0]))
        np.save(str(x) + ".npy", image)
        rawCamera.truncate(0)
        conn.send(data)
        x += 1
        continue
        
    if data == b'6':
        ser.write(data)
        camera.capture(rawCamera, format="bgr")
        image = rawCamera.array
        imagefinal = np.asarray(list(image[:,:,0] + [0, 0, 1]))
        np.save(str(x) + ".npy", image)
        rawCamera.truncate(0)
        conn.send(data)
        x += 1
        continue
        
    if data == b'7':
        ser.write(b'0')
        ser.close()
        time.sleep(5)
        conn.close()
        time.sleep(5)
        print('The server has been closed!')
        break
        
