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
    max_x2 = tf.reduce_max(x, axis=1, name="max_x2")

  with open(os.path.join(test_dir, 'output_nodes.txt'), 'w') as fid:
    fid.write(max_x2.op.name)

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, "test_max_2", test_dir)
    np_max_x2 = max_x2.eval()
    save_idx(np_max_x2, os.path.join(test_dir, "output_max_x2.idx"))


if __name__ == "__main__":
  generate()
