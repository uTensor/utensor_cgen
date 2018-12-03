import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(scope='session', name='droput_graph_tuple')
def dropout_graph_tuple():
    graph = tf.Graph()
    with graph.as_default():
        x = tf.constant(np.ones((5, 5)),
                        name='x', dtype=tf.float32)
        keep_prob = tf.placeholder(dtype=tf.float32,
                                   name='keep_prob')
        dropout_x = tf.nn.dropout(x, keep_prob, name='dropout_x')
        bias = tf.constant(0.5, name='bias', dtype=tf.float32)
        y = tf.add(dropout_x, bias, name='y')
    return (graph.as_graph_def(),
            [keep_prob.name, dropout_x.name],
            [y.op.name])
