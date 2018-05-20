import os

import numpy as np
import tensorflow as tf
from utensor_cgen.utils import save_consts, save_graph, save_idx


def generate():
  """Test SAME paddding

  pad_top == 0
  pad_left == 0
  larger floats
  """
  test_dir = os.path.abspath(os.path.dirname(__file__))
  graph = tf.Graph()
  with graph.as_default():
    input_data = np.arange(0, 1, 1 / (3 * 4 * 4 * 5)).reshape((3, 4, 4, 5))
    x = tf.constant(input_data, dtype=tf.float32, name="x")
    pool3 = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool3')

  with open(os.path.join(test_dir, 'output_nodes.txt'), 'w') as fid:
    fid.write(pool3.op.name)

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, 'test_max_pool_3', test_dir)
    np_output = pool3.eval()
    save_idx(np_output, os.path.join(test_dir, 'output_max_pool_3.idx'))


if __name__ == '__main__':
  generate()
