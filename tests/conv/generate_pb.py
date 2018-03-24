# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf


def generate():
  test_dir = os.path.dirname(__file__)
  np.random.seed(1234)
  graph = tf.Graph()
  with graph.as_default():
    x = tf.constant(np.random.random((3, 5, 5, 3)),
                    dtype=tf.float32,
                    name="x")
    w_filter = tf.constant(np.random.random((3, 3, 3, 2)), 
                           dtype=tf.float32,
                           name="w_filter")
    out_conv = tf.nn.conv2d(x, w_filter, 
                            strides=[1, 2, 2, 1], 
                            padding='VALID', 
                            name="out_conv")
  
  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, "test_conv", test_dir)
    np_out = sess.run(out_conv)
    save_idx(np_out, os.path.join(test_dir, "output_conv.idx"))


if __name__ == "__main__":
  generate()
