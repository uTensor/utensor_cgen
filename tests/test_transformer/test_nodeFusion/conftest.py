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
  matcher_graph = tf.Graph()
  with matcher_graph.as_default():
    matcher_input0 = tf.placeholder(dtype=tf.float32,
                                   name='matcher_input0')
    matcher_input1 = tf.placeholder(dtype=tf.float32,
                                   name='matcher_input1')
    matcher_node_add0 = tf.add(matcher_input0, matcher_input1, name="matcher_node_add0")
    matcher_node_add1 = tf.add(matcher_node_add0, matcher_input1, name="matcher_node_add1")
  
  umatchergraph = uTensorGraph(matcher_graph.as_graph_def(), [matcher_node_add1.name])
  #subgraph_tuple = (subgraph, (subgraph_input0, subgraph_input1), (subgraph_node_add1))
#######
  dropin_graph = tf.Graph()
  with dropin_graph.as_default():
    dropin_input0 = tf.placeholder(dtype=tf.float32,
                                   name='dropin_input0')
    dropin_input1 = tf.placeholder(dtype=tf.float32,
                                   name='dropin_input1')
    dropin_node_add0 = tf.add(dropin_input0, dropin_input1, name="dropin_node_add0")
  udropin = uTensorGraph(dropin_graph.as_graph_def(), [dropin_node_add0.name])
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

  return (ugraph, umatchergraph, udropin, uexpected)