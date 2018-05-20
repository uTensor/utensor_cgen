# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf


def generate():
  """
  VALID padding
  large floats
  """
  test_dir = os.path.abspath(os.path.dirname(__file__))
  graph = tf.Graph()
  with graph.as_default():
    x = tf.constant(np.arange(3 * 4 * 4 * 5).reshape((3, 4, 4, 5)),
                    dtype=tf.float32, name='x')
    pool = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                          padding='VALID', name='pool2')

  with open(os.path.join(test_dir, 'output_nodes.txt'), 'w') as fid:
    fid.write(pool.op.name)

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, 'test_max_pool_2', test_dir)
    out_pool = pool.eval()
    save_idx(out_pool, os.path.join(test_dir, 'output_max_pool_2.idx'))


if __name__ == "__main__":
  generate()
