# -*- cCARLnetg: utf-8 -*-

'''
Michael Hall and Daniel Ellis

CARLnet
'''

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def CARLnet_fn(features, labels, mode):
  input_layer = tf.reshape(features["x"], [-1, 90, 160, 1])
  
  input_layer = tf.to_float(input_layer)
  
  '''where all the layers go'''
  
  conv1 = tf.layers.conv2d(
          inputs = input_layer,
          filters = 32,
          strides = 2,
          kernel_size = [4,4],
          padding="same",
          activation = tf.nn.relu,
          name = 'conv1_4x4')
  
  pool1 = tf.layers.max_pooling2d(inputs = conv1,
                                  strides = 2,
                                  pool_size = [3,3],
                                  name = 'pool1')
  
  norm1 = tf.nn.local_response_normalization(pool1)
  
  
  conv2 = tf.layers.conv2d(
          inputs = norm1,
          filters = 32,
          strides = 1,
          kernel_size = [3,3],
          padding = 'same',
          activation = tf.nn.relu,
          name = 'conv2_3x3')
  
  pool2 = tf.layers.max_pooling2d(inputs = conv2,
                                  strides = 1,
                                  pool_size = [3,3],
                                  name = 'pool2')
  norm2 = tf.nn.local_response_normalization(pool2)
  
  conv3 = tf.layers.conv2d(
          inputs = norm2,
          filters = 32,
          strides = 2,
          kernel_size = [4,4],
          padding="same",
          activation = tf.nn.relu,
          name = 'conv3_4x4')
  
  pool3 = tf.layers.max_pooling2d(inputs = conv3,
                                  strides = 2,
                                  pool_size = [3,3],
                                  name = 'pool3')
  
  norm3 = tf.nn.local_response_normalization(pool3) 
  
  print(norm3.shape)
  flat = tf.reshape(norm3, [-1, 4*9*32])
  
  dropout = tf.layers.dropout(inputs = flat,
                              rate = 0.5,
                              training=mode == tf.estimator.ModeKeys.TRAIN)
  
  logits = tf.layers.dense(inputs = dropout, units = 3)
  
  
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
  accuracy = tf.metrics.accuracy(labels, predictions = tf.one_hot(predictions["classes"], 3), name = "accuracy")
  
  print(accuracy)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=tf.one_hot(predictions["classes"], 3), name='acc')}
  # Calculate Loss (for both TRAIN and EVAL modes)
  
  
  loss = tf.losses.softmax_cross_entropy(labels, logits)

  #accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
  #loss_summary = tf.summary.scalar("Loss", loss)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum = 0.9)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def init_carlnet(model_dir):
      CARLnet_classifier = tf.estimator.Estimator(
      model_fn=CARLnet_fn, model_dir= model_dir)
      
      return CARLnet_classifier