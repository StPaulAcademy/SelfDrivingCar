# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:15:59 2017

By Michael Hall and Daniel Ellis 

Used to train networks designed by michael and test convolution sizes
"""

import numpy as np
import stopnet_networks as networks
from email_mich import send_email
import tensorflow as tf
import time
import os


#Version numbers for the network
VERSION_1 = 0
VERSION_2 = 3

#set base parameters for network
WIDTH = 160
HEIGHT = 90
LR = 1e-3
EPOCHS = 25

#initialize data varible for email logging
data = []

#Load data
train_data = np.load('stopnet_training_data.npy')

#split into train and validataion
train = train_data[:-50]
test = train_data[-50:]

#X is for images, Y is labels
X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = np.array([i[1] for i in train])

#same for validation
test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = np.array([i[1] for i in test])

with tf.Graph().as_default():
    t = time.time()
    model_name = 'stopnet_v{}_{}.model'.format(VERSION_1,  VERSION_2)
    model, description = networks.stopnet(WIDTH, HEIGHT, LR, 0, 3, 3, 3)
    model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric = True, shuffle=True, run_id=model_name)
    print("trained!" + description)
    acc = model.evaluate(X,Y)
    data.append([description, acc[0]])
    model.save(os.path.join(os.curdir, "models", model_name))

#assemble data and send email
msg = "<b> Stopnet Training Results </b> <br>" + "Number of Epochs run: " + str(EPOCHS) + "<br> Training time (seconds): " + str(time.time()-t) + "<br><br>"
for item in data:
    msg += str(item[0]) + "<br>Accuracy:  "
    msg += str(item[1]) + "<br><br>"

send_email("Stopnet Tests", msg)
print("email sent!")
