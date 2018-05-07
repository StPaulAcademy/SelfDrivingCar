#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:59:57 2018

@author: michael
"""

import tensorflow as tf
import numpy as np
import cv2


export_dir = '/home/michael/Documents/ODIN/1525655894'


def predict_img(img_num):
    colors = [[(255,0,0), (0,255,0)],[(0,0,255), (255, 255, 0)]]
    data = np.load('ODIN_data.npy')
    train = data
      
    features = np.array([i[0] for i in train]).reshape(-1, 160, 160, 3)
    labels = np.array([i[1] for i in train])
      
    eval_data =  features[img_num:img_num + 1]
    eval_labels = labels[img_num:img_num + 1]
    
    predict_fn = tf.contrib.predictor.from_saved_model(export_dir)
    
    predict_result = predict_fn({'x': eval_data})
    predict_result = predict_result['output'][0].reshape((2,2,8))
    
    x_offset = 0
    y_offset = -80
    for i in range(2):
        y_offset += 80
        x_offset = 0
        for j in range(2):
            cv2.rectangle(eval_data[0,:,:,:], (int((predict_result[i][j][1]*80+x_offset) - (160*predict_result[i][j][3])/2),
                                      int((predict_result[i][j][2]*80 + y_offset) - (160*predict_result[i][j][4])/2)),
                                     (int((predict_result[i][j][1]*80 + x_offset) + (160*predict_result[i][j][3])/2),
                                      int((predict_result[i][j][2]*80 + y_offset) + (160*predict_result[i][j][4])/2)), colors[i][j], 2)
            x_offset += 80
            print(i,j)
    
    print(eval_labels)
    
    
    print(predict_result[0][0][1:5])
    print(predict_result[0][1][1:5])
    print(predict_result[1][0][1:5])
    print(predict_result[1][1][1:5])
    
    cv2.imshow('a', cv2.cvtColor(eval_data[0,:,:,:], cv2.COLOR_YUV2BGR))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for i in range(10):
    predict_img(i)