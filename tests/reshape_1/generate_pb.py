# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf


def generate():
  """flatten (fail, since we did not have QuantizedReshape in uTensor)
  """
  test_dir = os.path.dirname(__file__)
  graph = tf.Graph()
  with graph.as_default():
    x = tf.constant(np.arange(10).reshape(2, 5),
                    dtype=tf.float32,
                    name='x')
    output_x = tf.reshape(x, shape=[-1], name='x_reshape')

  with open(os.path.join(test_dir, 'output_nodes.txt'), 'w') as fid:
    fid.write(output_x.op.name)

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, 'test_reshape_1', test_dir)
    np_output = output_x.eval()
    save_idx(np_output, os.path.join(test_dir, 'output_x.idx'))


if __name__ == "__main__":
  generate()
