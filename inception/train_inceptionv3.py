# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:15:59 2017

@author: 18MichaelRH
"""

import numpy as np
from inceptionv3 import inceptionv3

TRIAL = 4

VERSION_1 = 0
VERSION_2 = 5

WIDTH = 160
HEIGHT = 90
LR = 1e-3
EPOCHS = 20
MODEL_NAME = 'ATPnet-v{}-{}.model'.format(VERSION_1,  VERSION_2)

model = inceptionv3(WIDTH, HEIGHT, LR)

train_data = np.load('traindata-2-{}.npy'.format(TRIAL))

train = train_data[:-100]
test = train_data[-100:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = [i[1] for i in test]

print(np.shape(X), np.shape(Y), np.shape(test_x), np.shape(test_y))

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric = True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
model.load(MODEL_NAME)