import tensorflow as tf
import numpy as np
import pytest
from utensor_cgen.ir import uTensorGraph

#provide ((graph), (subgraph), (replacement), (expected fused graph))
#graph: (graph, input nodes, output nodes)
#subgraph: (graph, input nodes, output nodes)
#replacement: (graph, input nodes, output nodes)
#expected fused graph: (graph, input nodes, output nodes)

@pytest.fixture(scope='session', name='fusion_graph_tuple')
def fusion_graph_tuple():
  graph = tf.Graph()
  with graph.as_default():
    input0 = tf.placeholder(dtype=tf.float32,
                                   name='input0')
    input1 = tf.placeholder(dtype=tf.float32,
                                   name='input1')
    input2 = tf.placeholder(dtype=tf.float32,
                                   name='input2')
    node_add0 = tf.add(input0, input1, name="node_add0")
    node_add1 = tf.add(node_add0, input1, name="node_add1")
    node_add2 = tf.add(node_add1, input2, name="node_add2")

  ugraph = uTensorGraph(graph.as_graph_def(), [node_add2.name])
  #graph_tuple = (graph, (input0, input1), (node_add2))
#######
  subgraph = tf.Graph()
  with subgraph.as_default():
    subgraph_input0 = tf.placeholder(dtype=tf.float32,
                                   name='subgraph_input0')
    subgraph_input1 = tf.placeholder(dtype=tf.float32,
                                   name='subgraph_input1')
    subgraph_node_add0 = tf.add(subgraph_input0, subgraph_input1, name="subgraph_node_add0")
    subgraph_node_add1 = tf.add(subgraph_node_add0, subgraph_input1, name="subgraph_node_add1")
  
  usubgraph = uTensorGraph(subgraph.as_graph_def(), [subgraph_node_add1.name])
  #subgraph_tuple = (subgraph, (subgraph_input0, subgraph_input1), (subgraph_node_add1))
#######
  replacement_graph = tf.Graph()
  with replacement_graph.as_default():
    replacement_input0 = tf.placeholder(dtype=tf.float32,
                                   name='replacement_input0')
    replacement_input1 = tf.placeholder(dtype=tf.float32,
                                   name='replacement_input1')
    replacement_node_add0 = tf.add(replacement_input0, replacement_input1, name="replacement_node_add0")
  ureplacement = uTensorGraph(replacement_graph.as_graph_def(), [replacement_node_add0.name])
  #replacement_tuple = (replacement_graph, (replacement_input0, replacement_input1), (replacement_node_add0))
#######
  expected_graph = tf.Graph()
  with expected_graph.as_default():
    expected_input0 = tf.placeholder(dtype=tf.float32,
                                   name='expected_input0')
    expected_input1 = tf.placeholder(dtype=tf.float32,
                                   name='expected_input1')
    expected_input2 = tf.placeholder(dtype=tf.float32,
                                   name='expected_input2')
    expected_node_add0 = tf.add(expected_input0, expected_input1, name="expected_node_add0")
    expected_node_add1 = tf.add(expected_node_add0, expected_input2, name="expected_node_add1")
  uexpected = uTensorGraph(expected_graph.as_graph_def(), [expected_node_add1.name])
  #expected_tuple = (expected_graph, (expected_input0, expected_input1), (expected_node_add0))

  return (ugraph, usubgraph, ureplacement, uexpected)