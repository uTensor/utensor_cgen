# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf


def generate():
  test_dir = os.path.dirname(__file__)
  graph = tf.Graph()
  with graph.as_default():
    x = tf.placeholder(tf.float32, name="x")
    one = tf.constant(1.0, dtype=tf.float32, name="one")
    y = tf.add(x, one, name="y")

  with open(os.path.join(test_dir, 'output_nodes.txt'), 'w') as fid:
    fid.write(y.op.name)

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, "test_placeholder", test_dir)


if __name__ == "__main__":
  generate()
