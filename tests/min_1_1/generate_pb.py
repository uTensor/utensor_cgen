# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf


def generate():
  test_dir = os.path.dirname(__file__)
  graph = tf.Graph()
  with graph.as_default():
    x = tf.constant(np.random.rand(10, 1), dtype=tf.float32, name="x")
    min_x = tf.reduce_min(x, name="min_x_1_1")

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, "test_min_1_1", test_dir)
    np_min_x = min_x.eval()
    save_idx(np_min_x, os.path.join(test_dir, "output_min_x1_1.idx"))


if __name__ == "__main__":
  generate()
