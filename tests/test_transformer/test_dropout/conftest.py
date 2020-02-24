from random import sample

import numpy as np
import pytest
import tensorflow as tf

from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.utils import random_str


@pytest.fixture(scope='session', name='droput_graph_tuple')
def dropout_graph_tuple():
    graph = tf.Graph()
    with graph.as_default():
        x = tf.constant(np.ones((5, 5)),
                        name='x', dtype=tf.float32)
        keep_prob = tf.placeholder(dtype=tf.float32,
                                   name='keep_prob')
        dropout_x = tf.nn.dropout(x, rate=1-keep_prob, name='dropout_x')
        bias = tf.constant(0.5, name='bias', dtype=tf.float32)
        y = tf.add(dropout_x, bias, name='y')
    return (graph.as_graph_def(),
            [keep_prob.name, dropout_x.name],
            [y.op.name])

@pytest.fixture(name='dropout_graph_tuple2')
def dropout_graph_tuple2():
    graph = tf.Graph()
    with graph.as_default():
        x = tf.constant(np.random.rand(10), dtype=tf.float32, name='x')
        rate = tf.placeholder(dtype=tf.float32, name='rate')
        drop = tf.nn.dropout(x, rate=rate, name=random_str(10))
    return graph.as_graph_def(), [drop.op.name]


@pytest.fixture(name='vgg_ugraph')
def gen_vgg_graph():
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(dtype=tf.float32, shape=[None, 2048, 2048, 3], name='input_x')
        rate = tf.placeholder(dtype=tf.float32, name='rate')
        in_feat = x
        num_layers = sample([3, 4, 5], 1)[0]
        for i in range(1, num_layers+1):
            ksize = sample([2, 3, 5], 1)[0]
            in_channel = in_feat.shape.as_list()[-1]
            out_channel = sample([3, 5, 10], 1)[0]
            stride = sample([1, 2], 1)[0]
            kernel = tf.constant(
                np.random.rand(ksize, ksize, in_channel, out_channel),
                dtype=tf.float32,
                name='kernel_{}'.format(i)
            )
            in_feat = tf.nn.conv2d(
                in_feat, 
                kernel,
                strides=[1, stride, stride, 1],
                padding='VALID',
                name='feat_map_{}'.format(i)
            )
            in_feat = tf.nn.relu(in_feat, name='relu_{}'.format(i))
            in_feat = tf.nn.max_pool(
                in_feat,
                ksize=[1, ksize, ksize, 1],
                strides=[1, stride, stride, 1],
                name='pool_{}'.format(i),
                padding='SAME',
            )
            if i != num_layers:
                in_feat = tf.nn.dropout(in_feat, rate=rate, name='dropout_{}'.format(i))   
    return GraphDefParser(config={}).parse(graph.as_graph_def(), output_nodes=[in_feat.op.name])
