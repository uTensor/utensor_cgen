# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf


def generate():
  test_dir = os.path.dirname(__file__)
  graph = tf.Graph()
  with graph.as_default():
    w = tf.constant(np.arange(1, 10).reshape((3, 3)), dtype=tf.float32, name="w")
    x = tf.constant(np.ones((3, 1)) / 3, dtype=tf.float32, name="x")
    z = tf.matmul(w, x, name='z')

  with open(os.path.join(test_dir, 'output_nodes.txt'), 'w') as fid:
    fid.write(z.op.name)

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, "test_matmul", test_dir)
    np_z = z.eval()
    save_idx(np_z, os.path.join(test_dir, "output_z.idx"))


if __name__ == "__main__":
  generate()
