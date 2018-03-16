# -*- coding: utf-8 -*-
"""
Michael Hall and Daniel Ellis

ODIN: Object Detection and Identification Network
"""

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def IOU(x1, y1, w1, h1, x2, y2, w2, h2): #Intersection over Union gives a metric of how close the bounding box is to the ground truth
    box11 = [ x1-(w1/2), y1-(h1/2) ]
    box12 = [ x1+(w1/2), y1+(h1/2) ]
    box21 = [ x2-(w2/2), y2-(h2/2) ]
    box22 = [ x2+(w2/2), y1+(h2/2) ]
    
    xI1 = tf.maximum(box11[0], box21[0])
    yI1 = tf.maximum(box11[1], box21[1])
    
    xI2 = tf.minimum(box12[0], box22[0])
    yI2 = tf.minimum(box12[1], box22[1])
    
    Iarea = (xI1 - xI2 + 1) * (yI1 - yI2 + 1)
    
    totalarea = w1*h1 + w2*h2
    
    Uarea = totalarea - Iarea
    
    return Iarea/Uarea

def ODINloss(logits, labels):
    #TERM1 Calculates error in center location of cells; labels[:, k, 0] is whether there is an object in the bounding box as a bool slicing because batch is first dimension
    labels = tf.cast(labels, tf.float32)

    term1 = 0
    k = 0
    for i in range(0,2):
        for j in range(0,2):
            term1 += labels[:, k, 0] * ((logits[:, i, j, 1]-labels[:, k, 1])**2 + (logits[:, i, j, 2]-labels[:, k, 2])**2)
            k += 1
    term1 = term1*5
    #log it
    tf.summary.scalar("Loss_term1", tf.reduce_mean(term1))
    #TERM2 Error in size of bounding boxes--square root to scale to size (small errors small box more important than small errors big box)
    term2 = 0
    k = 0
    for i in range(0,2):
        for j in range(0,2):
            term2 += labels[:, k, 0] * ( ((logits[:, i, j, 3])**(.5) - (labels[:, k, 3])**(.5))**2 +  ((logits[:, i, j, 4])**(.5) - (labels[:, k, 4])**(.5))**2 )
            k += 1
    term2 = term2 * 5
    
    tf.summary.scalar("Loss_term2", tf.reduce_mean(term2))
    #TERM3 error in confidence value--confidence is calculated as the IOU of the ground truth and predicted
    term3 = 0
    k = 0
    for i in range(0,2):
        for j in range(0,2):
            term3 += labels[:, k, 0] * (logits[:, i, j, 0] - IOU(logits[:, i, j, 1], logits[:, i, j, 2], logits[:, i, j, 3], logits[:, i, j, 4], labels[:, k, 1], labels[:, k, 2], labels[:, k, 3], labels[:, k, 4]))
            k += 1
    
    tf.summary.scalar("Loss_term3", tf.reduce_mean(term3))
    #TERM4 pushes confidence to 0 when there is no object in the quadrant
    term4 = 0
    k = 0
    for i in range(0,2):
        for j in range(0,2):
            term4 += ((-1 * labels[:, k, 0]) + 1) * (logits[:, i, j, 0] - IOU(logits[:, i, j, 1], logits[:, i, j, 2], logits[:, i, j, 3], logits[:, i, j, 4], labels[:, k, 1], labels[:, k, 2], labels[:, k, 3], labels[:, k, 4]))
            k += 1
    term4 = term4 * 0.5
    
    tf.summary.scalar("Loss_term4", tf.reduce_mean(term4))
    #TERM5 classification error using cross entropy
    term5 = 0
    k = 0
    class_labels = tf.cast(labels[:,k,5], tf.int32)
    for i in range(0,2):
        for j in range(0,2):
            term5 += tf.losses.sparse_softmax_cross_entropy(class_labels, logits[:, i, j, 5:])
            k += 1
    
    tf.summary.scalar("Loss_term5", tf.reduce_mean(term5))
    
    loss = term1 + term2 + term3 + term4 + term5

    loss = tf.reduce_mean(loss)
    tf.summary.scalar("loss", loss)
    return loss
    

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
          strides = 2,
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
          pool_size = [2,2],
          strides = 2,
          name = "logits")
  print(tf.shape(logits))
  
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      #"classes": tf.argmax(input=logits, axis=1),
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

#  # Add evaluation metrics (for EVAL mode)
#  eval_metric_ops = {
#      "accuracy": tf.metrics.accuracy(
#          labels=labels, predictions=predictions["classes"])}
#  return tf.estimator.EstimatorSpec(
#      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
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
  eval_data =  features
  eval_labels = labels

  # Create the Estimator
  ODIN_classifier = tf.estimator.Estimator(
      model_fn=ODIN_fn, model_dir="/tmp/ODIN_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
#  tensors_to_log = {"probabilities": "logits"}
#  logging_hook = tf.train.LoggingTensorHook(
#      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=128,
      num_epochs=100,
      shuffle=True)
  
  ODIN_classifier.train(
      input_fn=train_input_fn,
      steps=None)

  # Evaluate the model and print results
#  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#      x={"x": eval_data},
#      y=eval_labels,
#      num_epochs=1,
#      shuffle=False)
#  eval_results = ODIN_classifier.evaluate(input_fn=eval_input_fn)
#  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
