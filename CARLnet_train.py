# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:33:48 2018

Train CARLnet

Michael Hall and Daniel Ellis
"""

import CARLnet_v1 as CARLnet
import numpy as np
import tensorflow as tf

MODEL_DIR = "/tmp/carlnet_model"
EPOCHS = 10

data = np.load('traindata-2-4.npy')
train = data
print(np.shape(train[0]))
  
features = np.array([i[0] for i in train]).reshape(-1, 90, 160, 1)
labels = np.array([i[1] for i in train])
  
print(labels.shape)
print(np.shape(features))
  
  
train_data =  features[:-100]
train_labels = labels[:-100]
eval_data =  features[-100:]
eval_labels = labels[-100:]

CARLnet = CARLnet.init_carlnet(MODEL_DIR)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=128,
      num_epochs=EPOCHS,
      shuffle=True)
  
CARLnet.train(
      input_fn=train_input_fn,
      steps=None
      )