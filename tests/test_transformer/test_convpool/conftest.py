from random import sample

import numpy as np
import tensorflow as tf
from pytest import fixture

from utensor_cgen.frontend.tensorflow import GraphDefParser


@fixture(name='vgg_ugraph')
def gen_vgg_graph():
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(dtype=tf.float32, shape=[None, 2048, 2048, 3], name='input_x')
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
        ugraph = GraphDefParser(config={}).parse(graph.as_graph_def(), output_nodes=[in_feat.op.name])
    return ugraph
