# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util


def generate():
  test_dir = os.path.dirname(__file__)
  X = np.stack([0.5 * np.random.randn(50), 0.1 * np.random.randn(50)], axis=1)
  bias = 0.5
  W = np.array([0.29, 0.18])
  Y = X.dot(W) + bias + np.random.randn(50)*0.01

  train_graph = tf.Graph()
  with train_graph.as_default():
    tf_x = tf.constant(X, dtype=tf.float32, name="X")
    tf_y = tf.constant(Y, dtype=tf.float32, name="Y")

    tf_w = tf.Variable(np.random.randn(2, 1), dtype=tf.float32, name="W")
    tf_b = tf.Variable(0, dtype=tf.float32, name="b")
    tf_yhat = tf.add(tf.matmul(tf_x, tf_w), tf_b, name="yhat")
    loss = tf.reduce_mean(tf.pow(tf_yhat - tf_y, 2), name="loss")

    train_op = tf.train.AdamOptimizer(0.001).minimize(loss, name="train_op")

  with tf.Session(graph=train_graph) as sess:
    tf.global_variables_initializer().run()
    for step in range(1, 10001):
      _, l = sess.run([train_op, loss])
      if step % 1000 == 0:
        print("step:", step)
        print("loss:", l)
    const_graphdef = graph_util.convert_variables_to_constants(sess,
                                                               train_graph.as_graph_def(),
                                                               ["yhat"])
    graph = tf.Graph()
    with graph.as_default():
      tf.import_graph_def(const_graphdef, name='')
    with tf.Session(graph=graph) as sess:
      save_consts(sess, test_dir)
      save_graph(graph, "test_linreg", test_dir)
      pred = graph.get_tensor_by_name("yhat:0")
      np_pred = pred.eval()
      save_idx(np_pred, os.path.join(test_dir, "output_yhat.idx"))


if __name__ == "__main__":
  generate()
