import numpy as np
from pytest import fixture

import tensorflow as tf
from utensor_cgen.frontend.tensorflow import GraphDefParser


@fixture(name='patrn_fc_1')
def fully_connect_pattern1():
    patrn_graph = tf.Graph()
    with patrn_graph.as_default():
        z_prime = tf.placeholder(name='z_prime', dtype=tf.float32)
        w_prime = tf.constant(np.random.rand(3, 3), name='w_prime', dtype=tf.float32)
        a_prime = tf.matmul(z_prime, w_prime, name='a_prime')
        r_prime = tf.nn.relu(a_prime, name='r_prime')
    patrn_ugraph = GraphDefParser.parse(patrn_graph.as_graph_def(), output_nodes=[r_prime.op.name])
    for _ in range(2):
        patrn_ugraph.ops_info['z_prime'].add_null_input_tensor()
    return patrn_ugraph


@fixture(name='subj_graph_1')
def subject_ugraph_1():
    subj_graph = tf.Graph()
    with subj_graph.as_default():
        x = tf.constant(np.random.rand(3, 3), name='x', dtype=tf.float32)
        y = tf.constant(np.random.rand(3, 3), name='y', dtype=tf.float32)
        z = tf.add(x, y, name='z')
        w = tf.constant(np.random.rand(3, 3), name='w', dtype=tf.float32)
        a = tf.matmul(z, w, name='a')
        r = tf.nn.relu(a, name='r')
        out = tf.add(x, r, name='out')
    subj_ugraph = GraphDefParser.parse(subj_graph.as_graph_def(), output_nodes=[out.op.name])
    return subj_ugraph
