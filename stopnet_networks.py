#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:39:35 2017

Michael Hall and Daniel Ellis
"""
import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

def stopnet(width, height, lr, logging, conv1, conv2, conv3):
    network = input_data(shape=[None, width, height, 1], name='input')
    network = conv_2d(network, 32, conv1, strides=2, activation='relu', name='conv_5x5_1')
    network = max_pool_2d(network, 3, strides=2, name='pool_1')
    network = local_response_normalization(network)
    
    network = conv_2d(network, 32, conv2, strides=1, activation='relu', name='conv_3x3_1')
    network = max_pool_2d(network, 3, strides=1, name='pool_2')
    network = local_response_normalization(network)
    
    network = conv_2d(network, 32, conv3, strides=2, activation='relu', name='conv_3x3_2')
    network = max_pool_2d(network, 3, strides=2, name='pool_3')
    network = local_response_normalization(network)
    
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')
    model = tflearn.DNN(network, checkpoint_path = os.path.join(os.curdir, 'ckpt', 'net_v0_0_1'), max_checkpoints=1, tensorboard_verbose=logging)
    
    description = "Convolution Layer Setup: " + str(conv1) + ", " + str(conv2) + ", " + str(conv3)
    
    return model, description