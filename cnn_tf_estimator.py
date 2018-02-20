# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:44:38 2018

@author: 18MichaelRH
"""

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def carlnet_fn(features, labels, mode):
  input_layer = tf.reshape(features["x"], [-1, 90, 160, 1])
  
  input_layer = tf.to_float(input_layer)
  
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  print('pool shape')
  print(tf.shape(pool2))
  pool2_flat = tf.reshape(pool2, [-1, 40 * 22 * 64])
  
  print('pool_flat shape')
  print(tf.shape(pool2_flat))
  
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(inputs=dropout, units=3)
  print('logits shape')
  print(tf.shape(logits))
  
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.softmax_cross_entropy(labels, logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
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

  # Create the Estimator
  carlnet_classifier = tf.estimator.Estimator(
      model_fn=carlnet_fn, model_dir="/tmp/carlnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  carlnet_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = carlnet_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
