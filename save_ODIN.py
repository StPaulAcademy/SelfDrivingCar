#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 17:19:18 2018

@author: michael
"""

import tensorflow as tf
import numpy as np
import ODIN_v0_2
import ODIN as ODINv1


#ODIN = ODIN_v0_2.init_ODIN("model_2", 1)
ODIN = ODINv1.init_ODIN("model_2")


def serving_input_reciever_fn():
    inputs = {'x': tf.placeholder(shape = [None, 160, 160, 3], dtype = tf.float32)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
    
export_dir = ODIN.export_savedmodel(export_dir_base = '/home/michael/Documents/ODIN/', serving_input_receiver_fn = serving_input_reciever_fn)