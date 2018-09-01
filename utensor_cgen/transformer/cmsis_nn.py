# -*- coding:utf8 -*-
r"""CMSIS-NN Transformer

Node fusion and replacement for CMSIS-NN

"""
import re
from collections import defaultdict
from copy import deepcopy

import tensorflow as tf
import numpy as np

from utensor_cgen.ir import OperationInfo, uTensorGraph
from utensor_cgen.utils import parse_tensor_name

from .base import Transformer

__all__ = ["CMSIS_NN_Transformer"]

class uTensorNodeMeta(object):
  input_str = None
  node_name = None
  node_type = None
  node_attrib = None

  def __init__(self, str):
    tfNameMatcher = re.match( r'(.*)__uTensor__(.*[0-9]*)', str)
    self.node_attrib = None
    self.input_str = str

    if tfNameMatcher == None:
      cmd = "None"
      self.node_name = str
      return

    self.node_name = tfNameMatcher.group(1)

    if tfNameMatcher.group(2) == "End":
      cmd = "End"
    elif tfNameMatcher.group(2) == "Any":
      cmd = "Any"

def get_ops_io_info(op_type):
#please refer to OperatorFactory() in operators.py
  ops_io_table = dict()
  ops_io_table["Add"] =                 [[0, 0], [0]]
  ops_io_table["ArgMax"] =              [[0, 1], [0]]
  ops_io_table["Dequantize"] =          [[0, 1, 2], [0]]
  ops_io_table["Max"] =                 [[0, 1], [0]]
  ops_io_table["QuantizedMaxPool"] =    [[0, 1, 2], [0, 1, 2]]
  ops_io_table["Min"] =                 [[0, 1], [0]]
  ops_io_table["QuantizeV2"] =          [[0, 1, 2], [0, 1, 2]]
  ops_io_table["QuantizedMatMul"] =     [[0, 1, 2, 3, 4, 5], [0, 1, 2]]
  ops_io_table["QuantizedRelu"] =       [[0, 1, 2], [0, 1, 2]]
  ops_io_table["QuantizedAdd"] =        [[0, 0, 1, 2, 4, 5], [0, 1, 2]]
  ops_io_table["RequantizationRange"] = [[0, 1, 2], [0, 1]]
  ops_io_table["Requantize"] =          [[0, 1, 2, 3, 4], [0, 1, 2]]
  ops_io_table["Reshape"] =             [[0, 1], [0]]
  ops_io_table["QuantizedReshape"] =    [[0, 1, 2, 3], [0, 1, 2]]
  ops_io_table["QuantizedConv2D"] =     [[0, 1, 2, 3, 4, 5], [0, 1, 2]]
  ops_io_table["Const"] =               [None, [0]]
  ops_io_table["Placeholder"] =         [None, [0]]
  ops_io_table["Inline"] =              [None, [0]]

  return ops_io_table[op_type]


def get_input_nodes(graph, node_name):
  tensors_in = set([t.name for t in graph.ops_info[node_name].input_tensors])
  node_list = set()
  for it_node in graph.topo_order:
    if(it_node == node_name):
      continue
    it_tensors_out = [t.name for t in graph.ops_info[it_node].output_tensors]
    if not tensors_in.isdisjoint(it_tensors_out):
      node_list.add(it_node)

  return node_list

def get_output_nodes(graph, node_name):
  tensors_out = set([t.name for t in graph.ops_info[node_name].output_tensors])
  node_list = set()
  for it_node in graph.topo_order:
    if(it_node == node_name):
      continue
    it_tensors_in = [t.name for t in graph.ops_info[it_node].input_tensors]
    if not tensors_out.isdisjoint(it_tensors_in):
      node_list.add(it_node)

  return node_list

def is_connected(graph, node0, node1):
  input_nodes = get_input_nodes(graph, node0)
  output_nodes = get_output_nodes(graph, node0)
  node_list = input_nodes.union(output_nodes)

  return node1 in node_list

def get_input_tensors(graph, node_name):
  return [t.name for t in graph.ops_info[node_name].input_tensors]

def get_output_tensors(graph, node_name):
  return [t.name for t in graph.ops_info[node_name].output_tensors]

def subgraph_trace_exposed_edges(graph, start_index=0, end_index=None):
  if end_index == None:
    end_index = len(graph.topo_order)

  sub_topo = graph.topo_order[start_index:end_index]

  subgraph_tensors_in = set()
  subgraph_tensors_out = set()

  for node in sub_topo:
    subgraph_tensors_in.update(get_input_tensors(graph, node))
    subgraph_tensors_out.update(get_output_tensors(graph, node))

  input_edges = subgraph_tensors_in.difference(subgraph_tensors_out)
  output_edges = subgraph_tensors_out.difference(subgraph_tensors_in)

  input_edges_list = list()
  output_edges_list = list()

  #ensure this follows topological order
  for node in sub_topo:
    for t_name in get_input_tensors(graph, node):
      if t_name in input_edges:
        input_edges_list.append(t_name)
    for t_name in get_output_tensors(graph, node):
      if t_name in output_edges:
        output_edges_list.append(t_name)

  return [input_edges_list, output_edges_list]

def subgraph_trace_internal_edges(graph, start_index=0, end_index=None):
  if end_index == None:
    end_index = len(graph.topo_order)

  sub_topo = graph.topo_order[start_index:end_index]

  subgraph_tensors_in = set()
  subgraph_tensors_out = set()

  for node in sub_topo:
    subgraph_tensors_in.update(get_input_tensors(graph, node))
    subgraph_tensors_out.update(get_output_tensors(graph, node))
    
  internal_edges = subgraph_tensors_in.intersection(subgraph_tensors_out)

  internal_edges_list = list()

  #ensure this follows topological order
  for node in sub_topo:
    for t_name in get_input_tensors(graph, node):
      if t_name in internal_edges and not t_name in internal_edges_list:
        internal_edges_list.append(t_name)
    for t_name in get_output_tensors(graph, node):
      if t_name in internal_edges and not t_name in internal_edges_list:
        internal_edges_list.append(t_name)

  return internal_edges_list


def tensorInfo_from_name(graph, edge_name):
  for it_node in graph.topo_order:
    for t in graph.ops_info[it_node].input_tensors:
      if t.name == edge_name:
        return t
    for t in graph.ops_info[it_node].output_tensors:
      if t.name == edge_name:
        return t

  return None

#find the origin and destinations of a tensor
def get_tensor_node_names(graph, t_name):
  start_nodes = list()
  end_nodes = list()

  for it_node in graph.topo_order:
    for t in graph.ops_info[it_node].input_tensors:
      if t.name == t_name:
        end_nodes.append(it_node)
    for t in graph.ops_info[it_node].output_tensors:
      if t.name == t_name:
        start_nodes.append(it_node)

  return [start_nodes, end_nodes]

#named sequence
def forward_path_tracer(graph, start_node_name, end_node_name, depth=-1):
  # start_node = graph.ops_info[start_node_name]
  # end_node = graph.ops_info[end_node_name]
  output_node_names = set()
  path_list = list()

  if start_node_name == end_node_name:
    tmp = list()
    tmp.append(start_node_name)
    path_list.append(tmp)
    return path_list
  
  if depth == -1:
    depth = len(graph.topo_order) - 1

  output_node_names = get_output_nodes(graph, start_node_name)
  if len(output_node_names) == 0 or depth <= 0:
    return None
  
  for output_node_name in output_node_names:
    forward_path_list = forward_path_tracer(graph, output_node_name, end_node_name, depth-1)
    if forward_path_list == None:
      continue
    for forward_path in forward_path_list:
      forward_path.insert(0, start_node_name) #list
      path_list.append(forward_path) #list of list

  if len(path_list) == 0:
    return None

  return path_list

#TODO: FIXME:
#this function only has heurstic comparator
def compare_paths(path_group0, path_group1, graph0, graph1):
  path_potential_matches = dict() #path0 to path1
  #pass alternator
  if len(path_group0) != len(path_group1):
    return False

  for i0, path0 in enumerate(path_group0):
    for i1, path1 in enumerate(path_group1):
      #length comparison
      if len(path0) != len(path1):
        continue

      type_compare_success = True
      #type comparison
      for node0_name, node1_name in zip(path0, path1):
        node0 = graph0.ops_info[node0_name]
        node1 = graph1.ops_info[node1_name]

        if node0.op_type != node1.op_type:
          type_compare_success = False
          break
      if type_compare_success:
        if not i0 in path_potential_matches:
          path_potential_matches[i0] = list()
        path_potential_matches[i0].append(i1)

  #TODO: add more checks here
  if len(path_potential_matches) != len(path_group0):
    return False

  for i in path_potential_matches:
    if len(path_potential_matches[i]) < 1:
      return False
  return True

#recursive helper
#returns False when mismatch
#returns node-name-relation if a match exist
def isomorphic_associativity_helper(subject_node_name, matcher_node_name, subject_graph, matcher_graph, depth=0, subject_trace=None, matcher_trace=None):
  #compare the current nodes
  subject_node = subject_graph.ops_info[subject_node_name]
  matcher_node = matcher_graph.ops_info[matcher_node_name]
  
  node_meta_data = uTensorNodeMeta(matcher_node_name)
  #print(matcher_node_name)
  #import pdb; pdb.set_trace()
  
  if subject_node.op_type != matcher_node.op_type and node_meta_data.node_type != "Any":
    return False

  #create and concate the named-relations
  matcher_to_subject_nodes = dict()
  matcher_to_subject_edges = dict()
  matcher_to_subject_nodes[matcher_node_name] = subject_node_name

  matcher_node = matcher_graph.ops_info[matcher_node_name]
  [input_groups, _] = get_ops_io_info(matcher_node.op_type)
  #dealing with end of terminations

  if depth <= 0 or len(matcher_node.output_tensors) == 0 or input_groups == None or node_meta_data.node_type == "End":
    if subject_trace != None and matcher_trace != None:
      subject_start_node_name = subject_trace[0]
      subject_path_group = forward_path_tracer(subject_graph, subject_node_name, subject_start_node_name, len(matcher_graph.topo_order)-1)

      matcher_start_node_name = matcher_trace[0]
      matcher_path_group = forward_path_tracer(matcher_graph, matcher_node_name, matcher_start_node_name, len(matcher_graph.topo_order)-1)

      #import pdb; pdb.set_trace()

      if not compare_paths(subject_path_group, matcher_path_group, subject_graph, matcher_graph):
        return False

    return [matcher_to_subject_nodes, matcher_to_subject_edges]

  used_input_id = set()

  for group in list(range(0,max(input_groups)+1)):
    #this loop refer to the matcher
    for input_id, group_id in enumerate(input_groups):
      if group_id != group:
        continue
      #currently iterating over the same group

      #sweeping all inputs with the current group_id
      #mark used when successful
      for sweeping_input_id, sweeping_group_id in enumerate(input_groups):
        if sweeping_group_id != group_id or sweeping_input_id in used_input_id:
          continue
        #sweep the subject input nodes
        sweeping_subject_input_tensor_names = get_input_tensors(subject_graph, subject_node_name)
        sweeping_subject_input_tensor_name = sweeping_subject_input_tensor_names[sweeping_input_id]
        [sweeping_subject_input_node_name, _] = get_tensor_node_names(subject_graph, sweeping_subject_input_tensor_name)

        matcher_input_tensor_names = get_input_tensors(matcher_graph, matcher_node_name)
        matcher_input_tensor_name = matcher_input_tensor_names[input_id]
        [matcher_input_node_name, _] = get_tensor_node_names(matcher_graph, matcher_input_tensor_name)

        if subject_trace == None:
          subject_trace = list()
        if matcher_trace == None:
          matcher_trace = list()

        subject_trace.append(subject_node_name)
        matcher_trace.append(matcher_node_name)

        probe = isomorphic_associativity_helper(sweeping_subject_input_node_name[0], matcher_input_node_name[0], subject_graph, matcher_graph, depth-1, subject_trace, matcher_trace)
        if probe == False:
          continue
        #a match is found here:
        used_input_id.add(sweeping_input_id)
        matcher_to_subject_nodes.update(probe[0])
        matcher_to_subject_edges[matcher_input_tensor_name] = sweeping_subject_input_tensor_name
        matcher_to_subject_edges.update(probe[1])
  print("subject node: ", subject_node.name)
  print("matcher node: ", matcher_node.name)
  print("used_input_id vs input group: ", used_input_id, input_groups)
  # import pdb; pdb.set_trace()

  if len(used_input_id) == len(input_groups):
    return [matcher_to_subject_nodes, matcher_to_subject_edges]
  
  return False

def isomorphic_match(subject_graph, matcher_graph):
  matcher_to_subject_nodes = dict()
  matcher_to_subject_edges = dict()
  matcher_output_node_names = set()

  #identify matcher output nodes
  [_, matcher_output_edges] = subgraph_trace_exposed_edges(matcher_graph)
  
  for matcher_output_edge in matcher_output_edges: 
    [tensor_origin, _] = get_tensor_node_names(matcher_graph, matcher_output_edge)
    matcher_output_node_names.add(tensor_origin[0])
  max_search_depth = len(matcher_graph.topo_order)
  partial_matcher_to_subject_nodes = None
  partial_matcher_to_subject_edges = None
  #support only signle matcher output node for now
  for subject_node_name in subject_graph.topo_order:
    if subject_node_name == "zscore":
      import pdb; pdb.set_trace()
    probe = isomorphic_associativity_helper(subject_node_name, next(iter(matcher_output_node_names)), subject_graph, matcher_graph, max_search_depth)
    if probe == False:
      continue
    [partial_matcher_to_subject_nodes, partial_matcher_to_subject_edges] = probe

  if partial_matcher_to_subject_nodes == None and partial_matcher_to_subject_edges == None:
    return False

  matcher_to_subject_nodes.update(partial_matcher_to_subject_nodes)
  matcher_to_subject_edges.update(partial_matcher_to_subject_edges)

  return [matcher_to_subject_nodes, matcher_to_subject_edges]

class CMSIS_NN_Transformer(Transformer):
  METHOD_NAME = 'cmsisnn'
  KWARGS_NAMESCOPE = '_utensor_cmsisnn'

  def get_matcher_graph(self):
    graph = tf.Graph()
    with graph.as_default():
    
      x = tf.placeholder(dtype=tf.float32, name='input__uTensor__Any')
      #FIXME: type error, placeholder -> const
      W_fc1 = tf.placeholder(dtype=tf.float32, name='weight')
      b_fc1 = tf.placeholder(dtype=tf.float32, name='bias')
      a_fc1 = tf.add(tf.matmul(x, W_fc1), b_fc1, name="zscore")

    return uTensorGraph(graph.as_graph_def(), [a_fc1.name])

  def transform(self, ugraph):
    matcher_ugraph = self.get_matcher_graph()
    result = isomorphic_match(ugraph, matcher_ugraph)
    #import pdb; pdb.set_trace()
    print(result)
    assert result != False
    return ugraph
