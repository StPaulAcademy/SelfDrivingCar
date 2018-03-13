# -*- coding: utf-8 -*-
"""
Created By Daniel Ellis and Michael Hall
"""

import ctypes
import time
import socket
    
class Car():
    def __init__(self, port):
        self.lastCode = 0
        self.recieved = True
        self.port = port
        self.host = ''
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        print('A server opened at host ' + socket.gethostname() + ' on port ' + str(self.port)  + '!')
        self.connection, self.address = self.server.accept()
        print(self.address[0] + ' has connected!')

    def getKeyboard(self):
        w = ctypes.windll.user32.GetKeyState(0x57) >> 12
        a = ctypes.windll.user32.GetKeyState(0x41) >> 13
        s = ctypes.windll.user32.GetKeyState(0x53) >> 14
        d = ctypes.windll.user32.GetKeyState(0x44) >> 15
        self.keyCode = w ^ a ^ s ^ d
        if self.keyCode != self.lastCode:
            self.connection.send(bytes(str(self.keyCode).encode('UTF-8')))
        self.lastCode = self.keyCode
        
    def closeServer(self):
        self.connection.send(b'20')
        time.sleep(3)
        self.connection.close()
        print("Server closed")
try:       
    carl = Car(1348)
    while True:
        carl.getKeyboard()

except KeyboardInterrupt:
    carl.closeServer()
