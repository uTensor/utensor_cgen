# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf


def generate():
  test_dir = os.path.dirname(__file__)
  graph = tf.Graph()
  with graph.as_default():
    x = tf.constant(np.random.rand(5, 3), dtype=tf.float32, name="x")
    arg_max = tf.argmax(x, axis=1, name='argmax')

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, "test_argmax", test_dir)
    np_argmax = arg_max.eval()
    save_idx(np_argmax, os.path.join(test_dir, "output_argmax.idx"))


if __name__ == "__main__":
  generate()
