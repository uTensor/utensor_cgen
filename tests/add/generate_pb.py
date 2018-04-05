# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf


def generate():
  test_dir = os.path.dirname(__file__)
  graph = tf.Graph()
  with graph.as_default():
    x = tf.constant(np.array([1, 2, 3, 4]), dtype=tf.float32, name="x")
    y = tf.constant(np.array([1, 1, 1, 1]), dtype=tf.float32, name="y")
    z = tf.add(x, y, name="z")

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, "test_add", test_dir)
    np_z = z.eval()
    save_idx(np_z, os.path.join(test_dir, "output_z.idx"))


if __name__ == "__main__":
  generate()
