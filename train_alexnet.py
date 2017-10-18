# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:15:59 2017

@author: 18MichaelRH
"""

import numpy as np
from alexnet import alexnet

WIDTH = 150
HEIGHT = 150
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'APTnet-{}-{}-{}-epochs.model'.format(LR, 'alexnet', EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)
print("model loaded")

train_data = np.load('trainingdata.npy')
print('loaded data')
print(np.shape(train_data[0]), train_data[0])

train = train_data[:-5]
test = train_data[-5:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]

print(Y[200])

print(np.shape(X), np.shape(Y))

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric = True, run_id=MODEL_NAME)

model.save(MODEL_NAME)