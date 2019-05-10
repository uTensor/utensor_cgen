import pytest

import tensorflow as tf
from utensor_cgen.frontend.tensorflow import GraphDefParser


@pytest.fixture(scope='function', name='patrn_ugraph')
def patrn_ugraph():
    graph = tf.Graph()
    with graph.as_default():
        ptrn_input0 = tf.placeholder(dtype=tf.float32, name='input0')
        ptrn_input1 = tf.placeholder(dtype=tf.float32, name='input1')
        ptrn_add0 = tf.add(ptrn_input0, ptrn_input1, name='add0')
        ptrn_out = tf.add(ptrn_add0, ptrn_input1, name='output')
    ugraph = GraphDefParser.parse(graph.as_graph_def(), [ptrn_out.op.name])
    ugraph.ops_info[ptrn_input0.op.name].add_null_input_tensor()
    return ugraph

@pytest.fixture(scope='function', name='subject_ugraph1')
def subject_ugraph1():
    graph = tf.Graph()
    with graph.as_default():
        sub_input0 = tf.constant([i for i in range(10)], name='sub_input0')
        sub_input1 = tf.constant([i for i in range(10)], name='sub_input1')
        sub_input2 = tf.constant([i for i in range(10)], name='sub_input2')
        sub_add0 = tf.add(sub_input0, sub_input1, name='sub_add0')
        sub_add1 = tf.add(sub_add0, sub_input1, name='sub_add1')
        sub_output = tf.add(sub_add1, sub_input2, name='sub_output')
    ugraph = GraphDefParser.parse(graph.as_graph_def(), [sub_output.op.name])
    ugraph.ops_info[sub_input0.op.name].add_null_input_tensor()
    return ugraph

@pytest.fixture(scope='function', name='subject_ugraph1_1')
def subject_ugraph1_1():
    graph = tf.Graph()
    with graph.as_default():
        sub_input0 = tf.constant([i for i in range(10)], name='sub_input0')
        sub_input1 = tf.constant([i for i in range(10)], name='sub_input1')
        sub_input2 = tf.constant([i for i in range(10)], name='sub_input2')
        # permute
        sub_add0 = tf.add(sub_input1, sub_input0, name='sub_add0')
        sub_add1 = tf.add(sub_add0, sub_input1, name='sub_add1')
        sub_output = tf.multiply(sub_add1, sub_input2, name='sub_output')
    ugraph = GraphDefParser.parse(graph.as_graph_def(), [sub_output.op.name])
    ugraph.ops_info[sub_input0.op.name].add_null_input_tensor()
    return ugraph

@pytest.fixture(scope='function', name='subject_ugraph1_2')
def subject_ugraph1_2():
    graph = tf.Graph()
    with graph.as_default():
        sub_input0 = tf.constant([i for i in range(10)], name='sub_input0')
        sub_input1 = tf.constant([i for i in range(10)], name='sub_input1')
        sub_input2 = tf.constant([i for i in range(10)], name='sub_input2')
        sub_add0 = tf.add(sub_input0, sub_input1, name='sub_add0')
        sub_add1 = tf.add(sub_input1, sub_add0, name='sub_add1')
        sub_output = tf.multiply(sub_add1, sub_input2, name='sub_output')
    ugraph = GraphDefParser.parse(graph.as_graph_def(), [sub_output.op.name])
    ugraph.ops_info[sub_input0.op.name].add_null_input_tensor()
    return ugraph
