# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:15:59 2017

By Michael Hall and Daniel Ellis 

Used to train networks designed by michael and test convolution sizes
"""

import numpy as np
import networks
from email_mich import send_email
import tensorflow as tf




TRIAL = 4 #For Data Loading

#Version numbers for the network
VERSION_1 = 0
VERSION_2 = 0
VERSION_3 = 1

#set base parameters for network
WIDTH = 160
HEIGHT = 90
LR = 1e-3
EPOCHS = 30

#initialize data varible for email logging
data = []

#Load data
train_data = np.load('traindata-2-{}.npy'.format(TRIAL))

#split into train and validataion
train = train_data[:-100]
test = train_data[-100:]

#X is for images, Y is labels
X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = np.array([i[1] for i in train])

#same for validation
test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = np.array([i[1] for i in test])

#Training Loop(s)
for conv1 in range(2, 6):
    for conv2 in range(2, 6):
        for conv3 in range(2, 6):
            with tf.Graph().as_default():
                model_name = 'net_v{}_{}_{}.model'.format(VERSION_1,  VERSION_2, VERSION_3)
                model, description = networks.test_net(WIDTH, HEIGHT, LR, 0, conv1, conv2, conv3)
                model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric = True, shuffle=True, run_id=model_name)
                print("trained!")
                acc = model.evaluate(X,Y)
                data.append([description, acc[0]])
                VERSION_3 += 1




#assemble data and send email
msg = "<b> Convolution Training Results </b> <br>" + "Number of Epochs run: " + str(EPOCHS) + "<br><br>"
for item in data:
    msg += str(item[0]) + "<br>Accuracy:  "
    msg += str(item[1]) + "<br><br>"

send_email("Convolution layer sizes", msg)
print("email sent!")
