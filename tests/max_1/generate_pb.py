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
    max_x1 = tf.reduce_max(x, name="max_x1")

  with open(os.path.join(test_dir, 'output_nodes.txt'), 'w') as fid:
    fid.write(max_x1.op.name)

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, "test_max_1", test_dir)
    np_max_x1 = max_x1.eval()
    save_idx(np_max_x1, os.path.join(test_dir, "output_max_x1.idx"))


if __name__ == "__main__":
  generate()
