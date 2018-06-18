import tensorflow as tf
import pytest

@pytest.fixture(scope='session', name='refgraph_tuple')
def refgraph():
    graph = tf.Graph()
    with graph.as_default():
        x = tf.constant(1, name='x', dtype=tf.float32)
        y = tf.constant(1, name='y', dtype=tf.float32)
        z = tf.add(x, y, name='z')
        w = tf.add(x, 2.0, name='w')
        k = tf.add(z, w, name='k')
    refcnt_ans = {
        x.op.name : [2],
        y.op.name : [1],
        z.op.name : [1],
        w.op.name : [1]
    }

    return graph.as_graph_def(), refcnt_ans, [k.op.name]
