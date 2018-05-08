#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:59:57 2018

@author: michael
"""

import tensorflow as tf
import numpy as np
import cv2
import time


export_dir = '/home/michael/Documents/ODIN/1525655894'
data = np.load('ODIN_eval.npy')
predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

def IOU(x1, y1, w1, h1, x2, y2, w2, h2): #Intersection over Union gives a metric of how close the bounding box is to the ground truth
    box11 = [x1, y1]
    box12 = [w1, h1]
    box21 = [x2, y2]
    box22 = [w2, h2]
    
    xI1 = np.maximum(box11[0], box21[0])
    yI1 = np.maximum(box11[1], box21[1])
    
    xI2 = np.minimum(box12[0], box22[0])
    yI2 = np.minimum(box12[1], box22[1])
    
    Iarea = (xI1 - xI2 + 1) * (yI1 - yI2 + 1)
    
    totalarea = (box12[0] - box11[0] + 1) * (box12[1] - box11[1] + 1) + (box22[0] - box21[0] + 1) * (box22[1] - box21[1] + 1)
    
    Uarea = totalarea - Iarea
    
    IOU = Iarea/Uarea
    
    return IOU

def predict_img(img_num):
    t = time.time()
    colors = [[(255,0,0), (0,255,0)],[(0,0,255), (255, 255, 0)]]

      
    features = np.array([i[0] for i in data]).reshape(-1, 160, 160, 3)
    labels = np.array([i[1] for i in data])
      
    eval_data =  features[img_num:img_num + 1]
    eval_labels = labels[img_num:img_num + 1][0]

    predict_result = predict_fn({'x': eval_data})
    predict_result = predict_result['output'][0].reshape((2,2,8))
    
    x_offset = 0
    y_offset = -80
    q = 0
    for i in range(2):
        y_offset += 80
        x_offset = 0
        for j in range(2):
            x1 = int((predict_result[i][j][1]*80+x_offset) - (160*predict_result[i][j][3])/2)
            y1 = int((predict_result[i][j][2]*80 + y_offset) - (160*predict_result[i][j][4])/2)
            x2 = int((predict_result[i][j][1]*80 + x_offset) + (160*predict_result[i][j][3])/2)
            y2 = int((predict_result[i][j][2]*80 + y_offset) + (160*predict_result[i][j][4])/2)
            if eval_labels[q][0] != 0.0:
                ex1 = int(eval_labels[q][1] * 80 + x_offset - eval_labels[q][3] * 80)
                ey1 = int(eval_labels[q][2] * 80 + y_offset - eval_labels[q][4] * 80)
                ex2 = int(eval_labels[q][1] * 80 + x_offset + eval_labels[q][3] * 80)
                ey2 = int(eval_labels[q][2] * 80 + y_offset + eval_labels[q][4] * 80)
                cv2.rectangle(eval_data[0,:,:,:], (x1, y1), (x2, y2), colors[i][j], 2)
                cv2.rectangle(eval_data[0,:,:,:], (ex1, ey1), (ex2, ey2), (255, 255, 255), 2)
#                print(x1, y1, x2, y2)
#                print(ex1, ey1, ex2, ey2)
                print(IOU(x1, y1, x2, y2, ex1, ey1, ex2, ey2))
            x_offset += 80
            q += 1
    cv2.imshow('a', cv2.cvtColor(eval_data[0,:,:,:], cv2.COLOR_YUV2BGR))    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('prediction time: ' + str(time.time()-t))
    return IOU(x1, y1, x2, y2, ex1, ey1, ex2, ey2)


avg = None

#for i in range(len(data)):
#    if i == 0:
#        avg = predict_img(i)
#    else:
#        avg = (predict_img(i) + avg)/2
#print(avg)
print(len(data))
predict_img(749)