import pytest
import tensorflow as tf

@pytest.fixture(name='id_graph_def_1')
def id_graph_def_1():
    graph = tf.Graph()
    with graph.as_default():
        x = tf.constant(1, name='x', dtype=tf.float32)
        id_x = tf.identity(x, name='id_x')
        y = tf.constant(2, name='y', dtype=tf.float32)
        id_y = tf.identity(y, name='id_y')
        z = tf.add(id_x, id_y, name='z')
    return graph.as_graph_def()

@pytest.fixture(name='id_graph_def_2')
def id_graph_def_2():
    graph = tf.Graph()
    with graph.as_default():
        x = tf.constant(1, name='x', dtype=tf.float32)
        id_x = tf.identity(x, name='id_x')
        y = tf.constant(2, name='y', dtype=tf.float32)
        id_y = tf.identity(y, name='id_y')
        w = tf.add(x, y, name='w')
        id_w = tf.identity(w, name='id_w')
        z = tf.multiply(id_w, id_y, name='z')
    return graph.as_graph_def()