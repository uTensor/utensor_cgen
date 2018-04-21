#Test case for Quantized Add
import os
from utensor_cgen.utils import save_consts, save_graph, save_idx
import numpy as np
import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework import graph_util as gu


def generate():
  test_dir = os.path.dirname(__file__)
  # test case for Add
  graph = tf.Graph()
  with graph.as_default():
    x = tf.constant(np.array([1, 2, 3, 4]), dtype=tf.float32, name="x")
    y = tf.constant(np.array([1, 1, 1, 1]), dtype=tf.float32, name="y")
    z = tf.add(x, y, name="z")

  with open(os.path.join(test_dir, 'output_nodes.txt'), 'w') as fid:
    fid.write(z.op.name)

  transformed_graph_def = None
  with tf.Session(graph=graph) as sess:
    # Force the quantization
    transformed_graph_def = TransformGraph(sess.graph_def, [],
                                           ["z"], ["quantize_weights", "quantize_nodes"])
  graph2 = tf.Graph()
  with tf.Session(graph=graph2) as sess:
    # Load the transformed graph into current context
    tf.import_graph_def(transformed_graph_def, name="")

    save_consts(sess, test_dir)
    save_graph(graph, "test_quantized_add", test_dir)
    save_graph(graph2, "test_quant_quantized_add", test_dir)
    z_1 = sess.graph.get_tensor_by_name(sess.graph.get_operation_by_name("z").outputs[0].name)
    np_z = z_1.eval()
    save_idx(np_z, os.path.join(test_dir, "output_z.idx"))


if __name__ == '__main__':
    generate()
