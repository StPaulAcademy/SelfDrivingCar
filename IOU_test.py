# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 09:04:38 2018

@author: 18MichaelRH
"""


import tensorflow as tf

def IOU(x1, y1, w1, h1, x2, y2, w2, h2): #Intersection over Union gives a metric of how close the bounding box is to the ground truth
    box11 = [ x1-(w1/2), y1-(h1/2) ]
    box12 = [ x1+(w1/2), y1+(h1/2) ]
    box21 = [ x2-(w2/2), y2-(h2/2) ]
    box22 = [ x2+(w2/2), y1+(h2/2) ]
    

    
    
    xI1 = tf.maximum(box11[0], box21[0])
    yI1 = tf.maximum(box11[1], box21[1])
    
    xI2 = tf.minimum(box12[0], box22[0])
    yI2 = tf.minimum(box12[1], box22[1])
    
    Iarea = (xI1 - xI2) * (yI1 - yI2)
    
    totalarea = w1*h1 + w2*h2
    
    Uarea = totalarea - Iarea
    
    return Iarea/Uarea

with tf.Session() as sess:
    rect_1 = tf.Variable([.7, .4, .5, .6])
    rect_2 = tf.Variable([.7, .4, .6, .7])
    sess.run(tf.global_variables_initializer())
    result = sess.run(IOU(rect_1[0], rect_1[1], rect_1[2], rect_1[3], rect_2[0], rect_2[1], rect_2[2], rect_2[3]))
    
    print(result)