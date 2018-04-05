# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf


def generate():
  test_dir = os.path.dirname(__file__)
  graph = tf.Graph()
  with graph.as_default():
    x = tf.constant(0.5 * np.random.randn(5, 5), name="x", dtype=tf.float32)
    relu = tf.nn.relu(x, name="relu")

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, "test_relu", test_dir)
    np_relu = relu.eval()
    save_idx(np_relu, os.path.join(test_dir, "output_relu.idx"))


if __name__ == "__main__":
  generate()
