import pytest
import tensorflow as tf


@pytest.fixture(scope='session', name='refgraph_tuple')
def refgraph():
    graph = tf.Graph()
    with graph.as_default():
        x = tf.constant(1, name='x', dtype=tf.float32)
        y = tf.constant(1, name='y', dtype=tf.float32)
        z = tf.add(x, y, name='z')
        w = tf.add(x, 2.0, name='w')
        k = tf.add(z, w, name='k')


    return graph.as_graph_def(), [k.op.name]