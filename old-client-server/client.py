# -*- coding: utf-8 -*-
#Created by Daniel Ellis

import tkinter as tk
import socket
import time

class Client:
    def __init__(self, master, ip, port):
        
        self.PORT = port
        self.HOSTNAME = ip
        self.master = master
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.HOSTNAME, self.PORT))
        
        self.master.bind('<KeyPress-w>', self.forward)
        self.master.bind('<KeyRelease-w>', self.center)
        
        self.master.bind('<KeyPress-q>', self.f_left)
        self.master.bind('<KeyRelease-q>', self.center)

        self.master.bind('<KeyPress-e>', self.f_right)
        self.master.bind('<KeyRelease-e>', self.center)

        self.master.bind('<KeyPress-s>', self.backward)
        self.master.bind('<KeyRelease-s>', self.center)
        
        self.master.bind('<KeyPress-a>', self.b_left)
        self.master.bind('<KeyRelease-a>', self.center)

        self.master.bind('<KeyPress-d>', self.b_right)
        self.master.bind('<KeyRelease-d>', self.center)
        
        self.quitButton = tk.Button(self.master, text="Quit", command=self.quit_client)
        self.quitButton.place(x=0, y=0)           
            
    def center(self, event):
        self.s.send(b'0')
        self.s.recv(1)
        time.sleep(.1)
        
    def forward(self, event):
        self.s.send(b'1')
        self.s.recv(1)
        time.sleep(.1)
        
    def f_right(self, event):
        self.s.send(b'2')
        self.s.recv(1)
        time.sleep(.1)
        
    def f_left(self, event):
        self.s.send(b'3')
        self.s.recv(1)
        time.sleep(.1)
            
    def backward(self, event):
        self.s.send(b'4')
        self.s.recv(1)
        time.sleep(.1)
        
    def b_right(self, event):
        self.s.send(b'5')
        self.s.recv(1)
        time.sleep(.1)
        
    def b_left(self, event):
        self.s.send(b'6')
        self.s.recv(1)
        time.sleep(.1)
        
    def quit_client(self):
        self.s.send(b'7')
        self.s.recv(1)
        self.s.close()
        self.master.destroy()

root = tk.Tk()
gui = Client(root, '192.168.1.101', 1337)
root.mainloop()
        
