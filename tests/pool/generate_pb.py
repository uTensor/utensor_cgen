# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf


def generate():
  test_dir = os.path.dirname(__file__)
  graph = tf.Graph()
  with graph.as_default():
    x = tf.constant(np.arange(3*4*4*5).reshape((3, 4, 4, 5)), dtype=tf.float32, name='x')
    pool = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool')

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, 'test_pool', test_dir)
    out_pool = pool.eval()
    save_idx(out_pool, os.path.join(test_dir, 'out_pool.idx'))


if __name__ == "__main__":
  generate()
