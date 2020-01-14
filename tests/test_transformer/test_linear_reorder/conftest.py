from pytest import fixture

import tensorflow as tf
from utensor_cgen.frontend.tensorflow import GraphDefParser


@fixture(name='subj_ugraph_1')
def subject_ugraph_1():
    graph = tf.Graph()
    with graph.as_default():
        input_1 = tf.placeholder(dtype=tf.float32, shape=[None, 512, 512, 10], name='input_1')
        relu_1 = tf.nn.relu(input_1, name='relu_1')
        max_pool_1 = tf.nn.max_pool(relu_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_1')
        input_2 = tf.placeholder(dtype=tf.float32, shape=[None, 512, 512, 10], name='input_2')
        relu_2 = tf.nn.relu(input_2, name='relu_2')
        max_pool_2 = tf.nn.max_pool(relu_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_2')
        output = tf.add(max_pool_1, max_pool_2, name='output')
    subj_ugraph = GraphDefParser.parse(graph.as_graph_def(), output_nodes=[output.op.name])
    return subj_ugraph
