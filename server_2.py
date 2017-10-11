
import socket
import time
import serial
import picamera
import csv


camera = picamera.PiCamera()

moves = open('moves.csv', 'w')
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
data2 = 0
while True:
    data = conn.recv(1)
#    if data == data2:
#        camera.capture("/mnt/usb/img/" + str(x) + ".jpg")
#        x += 1
#        moves.write(str(data2) + ', ' )
#        continue
    if not data: break
    if data == b'7':
        ser.write(b'0')
        ser.close()
        conn.close()
        moves.close()
        print('The server has been closed!')
        break
    else:
        ser.write(data)
        moves.writerow([str(data)])
        if data != "b'0'":
            camera.capture("/mnt/usb/img/" +str(x) + ".jpg")
            x += 1
        print(data, x)
#    data2 = data