# -*- coding: utf-8 -*-
"""
Michael Hall and Daniel Ellis

ODIN: Object Detection and Identification Network
"""

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def ODINloss(logits, labels):
    term1 = 0
    k = 0
    for i in range(0,2):
        for j in range(0,2):
            term1 += labels[k][0] * ((logits[i][j][1]-labels[k][1])**2 + (logits[i][j][2]-labels[k][2])**2)
            k += 1
    term1 = term1*5
    
    term2 = 0
    k = 0
    for i in range(0,2):
        for j in range(0,2):
            term2 += labels[k][0] * ( ((logits[i][j][3])**(.5) - (logits[k][3])**(.5))**2 +  ((logits[i][j][4])**(.5) - (logits[k][4])**(.5))**2)
            k += 1
    term2 = term2 * 5
    
    

def ODIN_fn(features, labels, mode):
  input_layer = tf.reshape(features["x"], [-1, 160, 160, 1])
  
  input_layer = tf.to_float(input_layer)
  
  conv1 = tf.layers.conv2d(
          inputs = input_layer,
          filters = 32,
          kernel_size = [3,3],
          padding = 'same',
          activation = tf.nn.relu,
          name = "conv1_3x3")
  
  pool1 = tf.layers.max_pooling2d(
          inputs = conv1,
          pool_size = [2,2],
          strides = 2,
          name = "pool1")
  norm1 = tf.nn.local_response_normalization(pool1)
  
  conv_2 = tf.layers.conv2d(
          inputs = norm1,
          filters = 16,
          kernel_size = [1,1],
          padding = 'same',
          activation = tf.nn.relu,
          name = "conv2_1x1")
  
  conv_3 = tf.layers.conv2d(
          inputs = conv_2,
          filters = 64,
          kernel_size = [3,3],
          padding = 'same',
          activation = tf.nn.relu,
          name = "conv3_3x3")
  
  pool2 = tf.layers.max_pooling2d(
          inputs = conv_3,
          pool_size = [2,2],
          stries = 2,
          name = "pool2")
  norm2 = tf.nn.local_response_normalization(pool2)
  
  conv_4 = tf.layers.conv2d(
          inputs = norm2,
          filters = 16,
          kernel_size = [1,1],
          padding = 'same',
          activation = tf.nn.relu,
          name = 'conv4_1x1')
  conv_5 = tf.layers.conv2d(
          inputs = conv_4,
          filters = 64,
          strides = 2,
          kernel_size = [3,3],
          activation = tf.nn.relu,
          name = 'conv5_3x3')
  pool3 = tf.layers.max_pooling2d(
          inputs = conv_5,
          pool_size = [2,2],
          strides = 2,
          name = "pool3")
  norm3 = tf.nn.local_response_normalization(pool3)
  
  conv_6 = tf.layers.conv2d(
          inputs = norm3,
          filters = 32,
          strides = 2,
          kernel_size = [3,3],
          activation = tf.nn.relu,
          name = "conv6_3x3")
  conv_7 = tf.layers.conv2d(
          inputs = conv_6,
          filters = 8,
          kernel_size = [1,1],
          activation = tf.nn.relu,
          name = "conv7_1x1")
  
  logits = tf.layers.max_pooling2d(
          inputs = conv_7,
          pool_size = [3,3],
          strides = 3,
          name = "logits")
  print(tf.shape(logits))
  
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": logits
  }
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = ODINloss(logits, labels)

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
  ODIN_classifier = tf.estimator.Estimator(
      model_fn=ODIN_fn, model_dir="/tmp/ODIN_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "logits"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  ODIN_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = ODIN_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
