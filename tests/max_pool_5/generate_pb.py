import os

import numpy as np
import tensorflow as tf
from utensor_cgen.utils import save_consts, save_graph, save_idx


def generate():
  """Test SAME paddding

  expecting small error
  """
  test_dir = os.path.abspath(os.path.dirname(__file__))
  graph = tf.Graph()
  np.random.seed(3690)
  with graph.as_default():
    input_data = np.random.randint(0, 256, (3, 4, 4, 5), dtype=np.uint8)
    x = tf.constant(input_data, dtype=tf.float32, name="x")
    pool5 = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool5')

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, 'test_max_pool_5', test_dir)
    np_output = pool5.eval()
    save_idx(np_output, os.path.join(test_dir, 'output_max_pool_5.idx'))


if __name__ == '__main__':
  generate()
