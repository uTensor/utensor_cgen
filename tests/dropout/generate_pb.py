# -*- coding: utf8 -*-
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf


def generate():
  test_dir = os.path.dirname(__file__)
  np.random.seed(1234)
  graph = tf.Graph()
  with graph.as_default():
    x = tf.constant(np.arange(100).reshape((5, 5, 4)),
                    dtype=tf.float32, name='x')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # real output node name: output_dropout/mul
    output = tf.nn.dropout(x, keep_prob, name='dropout_output')

  with open(os.path.join(test_dir, 'output_nodes.txt'), 'w') as fid:
    fid.write(output.op.name)

  with tf.Session(graph=graph) as sess:
    save_consts(sess, test_dir)
    save_graph(graph, "test_dropout", test_dir)
    np_out = sess.run(output, feed_dict={keep_prob: 1.0})
    save_idx(np_out, os.path.join(test_dir, "dropout_output.idx"))


if __name__ == "__main__":
  generate()
