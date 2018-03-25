# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf


def generate():
  test_dir = os.path.dirname(__file__)
  graph = tf.Graph()
  with graph.as_default():
    x = tf.constant(np.random.rand(2, 2, 2), dtype=tf.float32, name="x")
    min_x = tf.reduce_min(x, axis=1, name="min_x_2")

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, "test_min_2", test_dir)
    np_min_x = min_x.eval()
    save_idx(np_min_x, os.path.join(test_dir, "min_2/output_min_x2.idx"))


if __name__ == "__main__":
  generate()
