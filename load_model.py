# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:15:59 2017

@author: 18MichaelRH
"""

import numpy as np
from inceptionv3 import inceptionv3

TRIAL = 4

VERSION_1 = 0
VERSION_2 = 3

WIDTH = 160
HEIGHT = 90
LR = 1e-2

MODEL_NAME = 'APTnet-v{}-{}.model'.format(VERSION_1,  VERSION_2)

model = inceptionv3(WIDTH, HEIGHT, LR)

model.load(MODEL_NAME)

train_data = np.load('traindata-2-{}.npy'.format(TRIAL))
X = np.array([i[0] for i in train_data]).reshape(-1, WIDTH, HEIGHT, 1)

#print(np.shape(np.array([X[465]])))
#print(np.array([X[1]]))

prediction = model.predict(np.array([X[5648]]))[0]
print(prediction)
print(np.around(prediction))
print(train_data[5648][1])