# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:15:59 2017

Authors: Michael Hall and Daniel Ellis
Advanced Technology Projects 2017

train_alexnet.py
"""

import numpy as np
from alexnet import alexnet

MODEL_VERSION = 0.1
#trial is for training data, version is for model version
TRIAL = 0
SAMPLES = 648

WIDTH = 150
HEIGHT = 150
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'APTnet-alexnet-v{}.model'.format(MODEL_VERSION)
TRAINING_DATA = 'trainingdata-{}-{}.npy'.format(TRIAL, SAMPLES)

model = alexnet(WIDTH, HEIGHT, LR)
print("model loaded")

train_data = np.load(TRAINING_DATA)
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