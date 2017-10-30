# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:16:56 2017

Michael Hall and Daniel Ellis

for use on computer to control pi
"""

import tkinter as tk
import socket
import time
import csv


class Server:
    def __init__(self, master, port):
        #initialize csv

        self.moves = open('moves.csv', 'w', newline='')
        self.moves_csv = csv.writer(self.moves)
        self.master = master
        #server setup
        self.HOST = ''
        self.PORT = 1337
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.HOST, self.PORT))
        self.s.listen(1)
        self.ready = True
        time.sleep(1)
        print('A server opened at host ' + socket.gethostname() + ' on port ' + str(self.PORT)  + '!')
        self.conn, self.addr = self.s.accept()
        print(self.addr[0] + ' has connected!')
        
        #accept connection
        
        
        #keybindings
        self.master.bind('<KeyPress-w>', self.forward)
        self.master.bind('<KeyRelease-w>', self.center)
        
        self.master.bind('<KeyPress-q>', self.f_left)
        self.master.bind('<KeyRelease-q>', self.center)

        self.master.bind('<KeyPress-e>', self.f_right)
        self.master.bind('<KeyRelease-e>', self.center)

#        self.master.bind('<KeyPress-s>', self.backward)
#        self.master.bind('<KeyRelease-s>', self.center)
#        
#        self.master.bind('<KeyPress-a>', self.b_left)
#        self.master.bind('<KeyRelease-a>', self.center)
#
#        self.master.bind('<KeyPress-d>', self.b_right)
#        self.master.bind('<KeyRelease-d>', self.center)
        
        self.quitButton = tk.Button(self.master, text="Quit", command=self.quit_client)
        self.quitButton.place(x=0, y=0)
        
    def center(self, event):
#        global last_stop
#        if time.time()-last_stop >= 0.25:
#            self.conn.send(b'0')
#            last_stop = time.time()
        self.conn.send(b'0')
        time.sleep(.01)
    def forward(self, event):
        self.update_ready()
        if self.ready == True:
            self.conn.send(b'1')
            self.moves_csv.writerow(str(1))
            time.sleep(.01)
    def f_right(self, event):
        self.update_ready()
        if self.ready == True:
            self.conn.send(b'2')
            self.moves_csv.writerow(str(2))
            time.sleep(.01)
    def f_left(self, event):
        self.update_ready()
        if self.ready == True:
            self.conn.send(b'3')
            self.moves_csv.writerow(str(3))
            time.sleep(.01)
    def backward(self, event):
        self.update_ready()
        if self.ready == True:
            self.conn.send(b'4')
            self.moves_csv.writerow(str(4))
            time.sleep(.01)
    def b_right(self, event):
        self.update_ready()
        if self.ready == True:
            self.conn.send(b'5')
            self.moves_csv.writerow(str(5))
            time.sleep(.01)
    def b_left(self, event):
        self.update_ready()
        if self.ready == True:
            self.conn.send(b'6')
            self.moves_csv.writerow(str(6))
            time.sleep(.01)
    def quit_client(self):
        self.conn.send(b'7')
        self.conn.close()
        self.master.destroy()
        
    def update_ready(self):
        self.tmp = self.conn.recv(1)
        if self.tmp == b'9':
            self.ready = True
        else:
            self.ready = False
              
root = tk.Tk()
last_stop = time.time()
gui = Server(root, 1337)
root.mainloop()