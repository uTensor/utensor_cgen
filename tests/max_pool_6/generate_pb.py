import os

import numpy as np
import tensorflow as tf
from utensor_cgen.utils import save_consts, save_graph, save_idx


def generate():
  """Test SAME paddding

  pad_left != 0
  pad_top != 0
  """
  test_dir = os.path.abspath(os.path.dirname(__file__))
  graph = tf.Graph()
  np.random.seed(3690)
  with graph.as_default():
    input_data = np.random.randint(0, 256, (3, 13, 13, 5))
    x = tf.constant(input_data, dtype=tf.float32, name="x")
    pool6 = tf.nn.max_pool(x, ksize=[1, 7, 7, 1], strides=[1, 3, 3, 1],
                           padding='SAME', name='pool6')

  with open(os.path.join(test_dir, 'output_nodes.txt'), 'w') as fid:
    fid.write(pool6.op.name)

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, 'test_max_pool_6', test_dir)
    np_output = pool6.eval()
    save_idx(np_output, os.path.join(test_dir, 'output_max_pool_6.idx'))


if __name__ == '__main__':
  generate()
