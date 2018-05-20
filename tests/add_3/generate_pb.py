# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf


def generate():
  """
  add test 3 (broadcasting)
  """
  test_dir = os.path.dirname(__file__)

  np.random.seed(1234)
  graph = tf.Graph()
  with graph.as_default():
    x = tf.constant(np.random.randn(3, 3), 
                    dtype=tf.float32,
                    name='x')
    b = tf.constant(1, dtype=tf.float32, name='b')
    z = tf.add(b, x, name='z')

  with open(os.path.join(test_dir, 'output_nodes.txt'), 'w') as fid:
    fid.write(z.op.name)

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, "test_add_3", test_dir)
    np_z = z.eval()
    save_idx(np_z, os.path.join(test_dir, "output_z.idx"))


if __name__ == "__main__":
  generate()
