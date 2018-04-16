#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 18:28:17 2018

@author: michael
"""
import tensorflow as tf
import ODIN_v0_2
import ODIN as ODINv1
import numpy as np

def main(unused_argv):
    
#    ODIN = ODIN_v0_2.init_ODIN("model_2", 0.5)
    ODIN = ODINv1.init_ODIN("model_1")
    # Load training and eval data
    data = np.load('ODINdata.npy')
    train = data
    print(np.shape(train[0]), train.shape)
  
  
    features = np.array([i[0] for i in train]).reshape(-1, 160, 160, 1)
    labels = np.array([i[1] for i in train])
  
  
    print(labels.shape)
    print(np.shape(features))
  
  
    train_data =  features
    train_labels = labels
    eval_data =  features[22:23]
    eval_labels = labels[22:23]

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
#  tensors_to_log = {"probabilities": "logits"}
#  logging_hook = tf.train.LoggingTensorHook(
#      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size= 128,
        num_epochs= 100,
        shuffle=True)
  
    ODIN.train(
        input_fn=train_input_fn,
        steps=None)
    
        
        
if __name__ == "__main__":
    tf.app.run()