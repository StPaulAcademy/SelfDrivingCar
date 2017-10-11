# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:14:44 2017

@author: 18MichaelRH
"""
import csv
import numpy as np

def write():
    csvfile = open("test.csv", 'w', newline='')
#        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)   
    writer = csv.writer(csvfile)
    for i in range(10):
        writer.writerow([i])
    
def read():
    moves = np.genfromtxt("test.csv", delimiter=',')
    print(moves)
write()
read()

