#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:59:57 2018

@author: michael
"""

import tensorflow as tf
import numpy as np
import ODIN_v0_2

export_dir = '/home/michael/Documents/ODIN/1523830894'

img_num = 64


data = np.load('ODINdata.npy')
train = data
  
features = np.array([i[0] for i in train]).reshape(-1, 160, 160, 1)
labels = np.array([i[1] for i in train])
  
eval_data =  features[img_num:img_num + 1]
eval_labels = labels[img_num:img_num + 1]

predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

predict_result = predict_fn({'x': eval_data})


print(eval_labels)

print(predict_result['output'][0].reshape((2,2,8)))