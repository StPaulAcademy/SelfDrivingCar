# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:52:19 2017

@author: 18DanielBE
"""

import serial
import tkinter as tk

ser = serial.Serial('COM4', 9600)
print('Serial Opened on BAUD 9600')
    
root = tk.Tk()

def forward(event):
    ser.write(b'1,255,1,0')
    print("Forward")
    
def backward(event):
    ser.write(b'1,255,0,0')
    print("Backward")
    
def stop(event):
    ser.write(b'1,0,1,0')

def lt(event):
    ser.write(b'0,0,0,45')
    print("Left")
    
def rt(event):
    ser.write(b'0,0,0,135')
    print("Right")
    
def center(event):
    ser.write(b'0,0,0,90')
    
def quit_client():
    ser.write(b'1,0,1,90')
    ser.write(b'0,0,1,90')
    print("Quitting")
    root.destroy()
    ser.close()
    
    
 
stop_client = tk.Button(root, text="Q", command=quit_client)
stop_client.place(x=20, y=30)
   
fwd = tk.Button(root, text="F")
fwd.place(x=20, y=0)
fwd.bind('<ButtonPress-1>', forward)
fwd.bind('<ButtonRelease-1>',stop)

bck = tk.Button(root, text="B")
bck.place(x=20, y=60)
bck.bind('<ButtonPress-1>', backward)
bck.bind('<ButtonRelease-1>',stop)

left = tk.Button(root, text="L")
left.place(x=0, y=30)
left.bind('<ButtonPress-1>', lt)
left.bind('<ButtonRelease-1>',center)

right = tk.Button(root, text="R")
right.place(x=44, y=30)
right.bind('<ButtonPress-1>', rt)
right.bind('<ButtonRelease-1>',center)

root.mainloop()