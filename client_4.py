# -*- coding: utf-8 -*-
"""
@author: Daniel Ellis
"""

import socket
import tkinter as tk
import time
from ctypes import windll


TCP_IP = '192.168.1.101' #raspberry pi ip address
TCP_PORT = 1340

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

def recieveByte(data):
    ready = False
    if s.recv(1) == data:
        ready = True
    return ready

while True:
    
    if windll.user32.GetKeyState(0x51) & 0x8000:
        #print('q')
        s.send(b'5')
        while recieveByte(b'5') == False:
            pass
        #time.sleep(.03)
        
    if windll.user32.GetKeyState(0x57) & 0x8000:
        #print('w')
        s.send(b'4')
        while recieveByte(b'4') == False:
            pass
        #time.sleep(.03)
        
    if windll.user32.GetKeyState(0x45) & 0x8000:
        #print('e')
        s.send(b'6')
        while recieveByte(b'6') == False:
            pass
        #time.sleep(.03)
        
    if windll.user32.GetKeyState(0x41) & 0x8000:
        #print('a')
        s.send(b'2')
        while recieveByte(b'2') == False:
            pass
        #time.sleep(.03)
        
    if windll.user32.GetKeyState(0x53) & 0x8000:
        #print('s')
        s.send(b'1')
        while recieveByte(b'1') == False:
            pass
        #time.sleep(.03)
        
    if windll.user32.GetKeyState(0x44) & 0x8000:
        #print('d')
        s.send(b'3')
        while recieveByte(b'3') == False:
            pass
        #time.sleep(.03)
        
    if windll.user32.GetKeyState(0x54) & 0x8000:
        print('Quitting')
        s.send(b'7')
        break
    
    else:
        #print('c')
        s.send(b'0')
        while recieveByte(b'0') == False:
            pass
        time.sleep(.03)
    
