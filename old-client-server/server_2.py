
import socket
import time
import serial
import picamera
from picamera.array import PiRGBArray
import csv
import numpy as np


camera = picamera.PiCamera()
rawCamera= PiRGBArray(camera)
time.sleep(2)

moves = open('moves.csv', 'w', newline='')
moves_csv = csv.writer(moves)

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
    ts = time.time()
    data = conn.recv(1)
    print('Receiving Data', time.time()-ts)
    if not data: 
        break
    if data == b'7':
        ser.write(b'0')
        ser.close()
        conn.close()
        moves.close()
        print('The server has been closed!')
        break
    else:
        ser.write(data)
        print('Sending Serial Data', time.time()-ts)
        if data != b'0':
            camera.capture(rawCamera, format="bgr")
            image = rawCamera.array
            np.save(str(x) + ".npy", image)
            moves_csv.writerow([str(data)[2]])
            x += 1
        rawCamera.truncate(0)
        print('Writing Image', time.time()-ts)
        conn.send(data)
        print('Sending Data', time.time()-ts)
        print(x, data)
