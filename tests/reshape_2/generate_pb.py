# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf


def generate():
  """the reshape op will be used in this case since tensorflow will flatten
  the input tensor and find the min/max value for quantized matmul
  """
  test_dir = os.path.dirname(__file__)
  graph = tf.Graph()
  with graph.as_default():
    x = tf.placeholder(tf.float32,
                       shape=[None, 3],
                       name='x')
    w = tf.constant(0.5 * np.random.randn(3, 1),
                    dtype=tf.float32,
                    name='w')
    y = tf.matmul(x, w, name='y')

  with open(os.path.join(test_dir, 'output_nodes.txt'), 'w') as fid:
    fid.write(y.op.name)

  np_x = 0.5 * np.random.randn(5, 3).astype(np.float32)
  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, 'test_reshape_2', test_dir)
    np_output = y.eval(feed_dict={'x:0': np_x})
    save_idx(np_x, os.path.join(test_dir, 'input_x.idx'))
    save_idx(np_output, os.path.join(test_dir, 'output_y.idx'))


if __name__ == "__main__":
  generate()
