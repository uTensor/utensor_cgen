import tensorflow as tf
import numpy as np
import pytest
from utensor_cgen.ir import uTensorGraph

#provide ((graph), (subgraph), (replacement), (expected fused graph))
#graph: (graph, input nodes, output nodes)
#subgraph: (graph, input nodes, output nodes)
#replacement: (graph, input nodes, output nodes)
#expected fused graph: (graph, input nodes, output nodes)

# helper functions
def make_rand_const(shape, name):
  val = np.random.random(shape)
  #return tf.constant(val, name)
  return tf.convert_to_tensor(val, name=name, dtype=tf.float32)

#tensorboard --logdir=./logs_graph0
def test_graph0():
  graph = tf.Graph()
  summary = tf.summary.FileWriter('./logs_graph0')
  with graph.as_default():
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    add_op = tf.add(x, x, name="inputAdd")
    W_fc1 = make_rand_const([784, 128], name='W_fc1')
    b_fc1 = make_rand_const([128], name='b_fc1')
    a_fc1 = tf.add(tf.matmul(add_op, W_fc1), b_fc1, name="zscore")
    h_fc1 = tf.nn.relu(a_fc1, name="act1")
    summary.add_graph(graph)
  
  ugraph = uTensorGraph(graph.as_graph_def(), ["act1"])
  return ugraph

#tensorboard --logdir=./logs_graph1 --port=8008
def test_graph1():
  graph1 = tf.Graph()
  summary = tf.summary.FileWriter('./logs_graph1')
  with graph1.as_default():
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    W_fc1 = make_rand_const([784, 128], name='W_fc1')
    b_fc1 = make_rand_const([128], name='b_fc1')
    a_fc1 = tf.add(tf.matmul(x, W_fc1), b_fc1, name="zscore1")
    h_fc1 = tf.nn.relu(a_fc1, "act1")
    
    W_fc2 = make_rand_const([128, 64], name='W_fc2')
    b_fc2 = make_rand_const([64], name='b_fc2')
    a_fc2 = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2, name="zscore2")
    h_fc2 = tf.nn.relu(a_fc2, "act2")
    summary.add_graph(graph1)

  ugraph1 = uTensorGraph(graph1.as_graph_def(), ["act2"])
  return ugraph1

@pytest.fixture(scope='session', name='fusion_graph_tuple')
def fusion_graph_tuple():
  graph0 = test_graph0()
  graph1 = test_graph1()
  return (graph0, graph1)
