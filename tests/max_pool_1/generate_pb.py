import os

import numpy as np
import tensorflow as tf
from utensor_cgen.utils import save_consts, save_graph, save_idx


def generate():
  test_dir = os.path.abspath(os.path.dirname(__file__))
  graph = tf.Graph()
  with graph.as_default():
    input_data = np.arange(0, 1.0, 1.0 / (3 * 10 * 10 * 5)).reshape((3, 10, 10, 5))
    x = tf.constant(input_data, dtype=tf.float32, name="x")
    pool1 = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, 'test_max_pool_1', test_dir)
    np_output = pool1.eval()
    save_idx(np_output, os.path.join(test_dir, 'output_pool.idx'))


if __name__ == '__main__':
  generate()
