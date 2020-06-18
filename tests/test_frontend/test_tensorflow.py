import numpy as np
import tensorflow.compat.v1 as tf


def test_scalar_shape():
    from utensor_cgen.frontend.tensorflow import GraphDefParser

    graph = tf.Graph()
    with graph.as_default():
        tf.constant(1, dtype=tf.float32, name='x')
    parser = GraphDefParser({})
    ugraph = parser.parse(graph.as_graph_def(), output_nodes=['x'])
    # shape of scalar tensor should be empty list
    out_tensor = ugraph.ops_info['x'].output_tensors[0]
    assert out_tensor.shape == []
    assert out_tensor.dtype is np.dtype('float32')

def test_placeholder_shape():
    from utensor_cgen.frontend.tensorflow import GraphDefParser

    graph = tf.Graph()
    with graph.as_default():
        tf.placeholder(dtype=tf.float32, name='x')
    parser = GraphDefParser({})
    ugraph = parser.parse(graph.as_graph_def(), output_nodes=['x'])
    # nondeterministic shape, can be any shape
    out_tensor = ugraph.ops_info['x'].output_tensors[0]
    assert out_tensor.shape is None
    assert out_tensor.dtype is np.dtype('float32')

    graph = tf.Graph()
    with graph.as_default():
        tf.placeholder(dtype=tf.float32, name='x', shape=[None, 5])
    parser = GraphDefParser({})
    ugraph = parser.parse(graph.as_graph_def(), output_nodes=['x'])
    # nondeterministic dimension
    out_tensor = ugraph.ops_info['x'].output_tensors[0]
    assert out_tensor.shape == [None, 5]
    assert out_tensor.dtype is np.dtype('float32')

def test_normal_tensor_shape():
    from utensor_cgen.frontend.tensorflow import GraphDefParser
    shape = np.random.randint(1, 10, size=(10,)).tolist()

    graph = tf.Graph()
    with graph.as_default():
        tf.constant(np.random.rand(*shape), dtype=tf.float32, name='x')
    parser = GraphDefParser({})
    ugraph = parser.parse(graph.as_graph_def(), output_nodes=['x'])
    # deterministic shape
    out_tensor = ugraph.ops_info['x'].output_tensors[0]
    assert out_tensor.shape == shape, 'expecting {}, get {}'.format(shape, out_tensor.shape)
    assert out_tensor.dtype is np.dtype('float32')
