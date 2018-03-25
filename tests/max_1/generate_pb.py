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
    max_x1_1 = tf.reduce_max(x, name="max_x1_1")

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, "test_max_1_1", test_dir)
    np_max_x1_1 = max_x1_1.eval()
    save_idx(np_max_x1_1, os.path.join(test_dir, "max_1_1/output_max_x1_1.idx"))


if __name__ == "__main__":
  generate()
