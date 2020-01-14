from random import sample

import numpy as np
from pytest import fixture

import tensorflow as tf
from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.transformer.pipeline import TransformerPipeline


@fixture(name='vgg_ugraph_pair', scope='function')
def gen_vgg_graph():
    graph = tf.Graph()
    trans = TransformerPipeline([
            'linear_reorder',
            'quantize',
    ])
    with graph.as_default():
        x = tf.placeholder(dtype=tf.float32, shape=[None, 2048, 2048, 3], name='input_x')
        in_feat = x
        num_layers = sample([3, 4, 5], 1)[0]
        num_layers = 3
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
        ugraph = GraphDefParser.parse(graph.as_graph_def(), output_nodes=[in_feat.op.name])
        ugraph = trans.transform(ugraph)
    return ugraph, num_layers
