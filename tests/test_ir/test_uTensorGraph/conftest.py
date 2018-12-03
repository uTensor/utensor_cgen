import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(scope='session', name='graph_tuple')
def simple_graph_and_outputs():
    graph = tf.Graph()
    with graph.as_default():
        weight = tf.constant(np.random.randn(3, 3), 
                             dtype=tf.float32,
                             name='weight')
        bias = tf.constant(np.pi,
                           dtype=tf.float32,
                           name='bias')
        x1 = tf.constant(np.random.randn(10, 3),
                         dtype=tf.float32,
                         name='x1')
        x2 = tf.add(tf.matmul(x1, weight), bias, name='x2')
        bias2 = tf.constant(3.69, name='bias2', dtype=tf.float32)
        x3 = tf.multiply(x2, bias2, name='x3')
    return graph.as_graph_def(), [x2.op.name, x3.op.name]
