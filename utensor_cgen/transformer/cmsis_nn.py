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
from tensorflow.tools.graph_transforms import TransformGraph
from utensor_cgen.ir.utils import graph_check

from .base import Transformer

__all__ = ["CMSIS_NN_Transformer"]

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
  ops_io_table["MatMul"] =              [[0, 1], [0]]
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


def get_input_node_names(graph, node_name):
  input_op_infos = graph.ops_info[node_name].input_nodes
  input_op_names = [op.name for op in input_op_infos]

  return input_op_names

def get_output_node_names(graph, node_name):
  output_op_infos = graph.ops_info[node_name].output_nodes
  output_op_names = [op.name for op in output_op_infos]

  return output_op_names

def is_connected(graph, node0, node1):
  input_nodes = get_input_node_names(graph, node0)
  output_nodes = get_output_node_names(graph, node0)
  node_list = input_nodes.union(output_nodes)

  return node1 in node_list

def get_input_tensor_names(graph, node_name):
  return [t.name for t in graph.ops_info[node_name].input_tensors]

def get_output_tensor_names(graph, node_name):
  return [t.name for t in graph.ops_info[node_name].output_tensors]

def subgraph_trace_exposed_edges(graph, start_index=0, end_index=None):
  if end_index == None:
    end_index = len(graph.topo_order)

  sub_topo = graph.topo_order[start_index:end_index]

  subgraph_tensors_in = set()
  subgraph_tensors_out = set()

  for node in sub_topo:
    subgraph_tensors_in.update(get_input_tensor_names(graph, node))
    subgraph_tensors_out.update(get_output_tensor_names(graph, node))

  input_edges = subgraph_tensors_in.difference(subgraph_tensors_out)
  output_edges = subgraph_tensors_out.difference(subgraph_tensors_in)

  input_edges_list = list()
  output_edges_list = list()

  #ensure this follows topological order
  for node in sub_topo:
    for t_name in get_input_tensor_names(graph, node):
      if t_name in input_edges:
        input_edges_list.append(t_name)
    for t_name in get_output_tensor_names(graph, node):
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
    subgraph_tensors_in.update(get_input_tensor_names(graph, node))
    subgraph_tensors_out.update(get_output_tensor_names(graph, node))
    
  internal_edges = subgraph_tensors_in.intersection(subgraph_tensors_out)

  internal_edges_list = list()

  #ensure this follows topological order
  for node in sub_topo:
    for t_name in get_input_tensor_names(graph, node):
      if t_name in internal_edges and not t_name in internal_edges_list:
        internal_edges_list.append(t_name)
    for t_name in get_output_tensor_names(graph, node):
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

  output_node_names = get_output_node_names(graph, start_node_name)
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


def get_node_meta(node_name, meta):
  if meta == None:
    return ["None"]
  if not node_name in meta:
    return ["None"]
  return meta[node_name]

#TODO: FIXME:
#this function only has heurstic comparator
def compare_paths(path_group0, path_group1, graph0, graph1, meta0=None, meta1=None):
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

        if node0.op_type != node1.op_type and (not "Any" in get_node_meta(node0_name, meta0) and not "Any" in get_node_meta(node1_name, meta1)):
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
def isomorphic_associativity_helper(subject_node_name, matcher_node_name, subject_graph, matcher_graph, matcher_meta=None, depth=0, subject_trace=None, matcher_trace=None):
  #compare the current nodes
  subject_node = subject_graph.ops_info[subject_node_name]
  matcher_node = matcher_graph.ops_info[matcher_node_name]
  
  if subject_node.op_type != matcher_node.op_type and not "Any" in get_node_meta(matcher_node_name, matcher_meta) :
    return False

  #create and concate the named-relations
  matcher_to_subject_nodes = dict()
  matcher_to_subject_edges = dict()
  matcher_to_subject_nodes[matcher_node_name] = subject_node_name

  matcher_node = matcher_graph.ops_info[matcher_node_name]
  [input_groups, _] = get_ops_io_info(matcher_node.op_type)
  #dealing with end of terminations

  if depth <= 0 or len(matcher_node.output_tensors) == 0 or input_groups == None or "End" in get_node_meta(matcher_node_name, matcher_meta):
    if subject_trace != None and matcher_trace != None:
      subject_start_node_name = subject_trace[0]
      subject_path_group = forward_path_tracer(subject_graph, subject_node_name, subject_start_node_name, len(matcher_graph.topo_order)-1)

      matcher_start_node_name = matcher_trace[0]
      matcher_path_group = forward_path_tracer(matcher_graph, matcher_node_name, matcher_start_node_name, len(matcher_graph.topo_order)-1)

      if not compare_paths(subject_path_group, matcher_path_group, subject_graph, matcher_graph, None, matcher_meta):
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
        sweeping_subject_input_tensor_names = get_input_tensor_names(subject_graph, subject_node_name)
        sweeping_subject_input_tensor_name = sweeping_subject_input_tensor_names[sweeping_input_id]
        [sweeping_subject_input_node_name, _] = get_tensor_node_names(subject_graph, sweeping_subject_input_tensor_name)

        matcher_input_tensor_names = get_input_tensor_names(matcher_graph, matcher_node_name)
        matcher_input_tensor_name = matcher_input_tensor_names[input_id]
        [matcher_input_node_name, _] = get_tensor_node_names(matcher_graph, matcher_input_tensor_name)

        if subject_trace == None:
          subject_trace = list()
        if matcher_trace == None:
          matcher_trace = list()

        subject_trace.append(subject_node_name)
        matcher_trace.append(matcher_node_name)

        probe = isomorphic_associativity_helper(sweeping_subject_input_node_name[0], matcher_input_node_name[0], subject_graph, matcher_graph, matcher_meta, depth-1, subject_trace, matcher_trace)
        if probe == False:
          continue
        #a match is found here:
        used_input_id.add(sweeping_input_id)
        matcher_to_subject_nodes.update(probe[0])
        matcher_to_subject_edges[matcher_input_tensor_name] = sweeping_subject_input_tensor_name
        matcher_to_subject_edges.update(probe[1])
        
  if len(used_input_id) == len(input_groups):
    return [matcher_to_subject_nodes, matcher_to_subject_edges]
  
  return False

def isomorphic_match(subject_graph, matcher_graph, meta):
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
    probe = isomorphic_associativity_helper(subject_node_name, next(iter(matcher_output_node_names)), subject_graph, matcher_graph, meta, max_search_depth)
    if probe == False:
      continue
    [partial_matcher_to_subject_nodes, partial_matcher_to_subject_edges] = probe

  if partial_matcher_to_subject_nodes == None and partial_matcher_to_subject_edges == None:
    return False

  matcher_to_subject_nodes.update(partial_matcher_to_subject_nodes)
  matcher_to_subject_edges.update(partial_matcher_to_subject_edges)

  return [matcher_to_subject_nodes, matcher_to_subject_edges]

def replace_tensors_op(node_name, new_node_name, graph):
  for op_name, op_info in graph.ops_info.items():
    for i, output_tensor_info in enumerate(op_info.output_tensors):
      if output_tensor_info.op_name == node_name:
        output_tensor_info.op_name = new_node_name
        op_info.output_tensors[i] = output_tensor_info
        graph.ops_info[op_name] = op_info

    for i, input_tensor_info in enumerate(op_info.input_tensors):
      if input_tensor_info.op_name == node_name:
        input_tensor_info.op_name = new_node_name
        op_info.input_tensors[i] = input_tensor_info
        graph.ops_info[op_name] = op_info

  return graph

def graph_validate(graph):
  conflicts = []  
  for op_name, op_info in graph.ops_info.items():
    for input_tensor_info in op_info.input_tensors:
      if input_tensor_info.op_name not in graph.ops_info:
        print("In %r: input tensor %r points to non-existing op %r" % (op_name, input_tensor_info.name, input_tensor_info.op_name))
        conflicts.append((input_tensor_info.name, input_tensor_info.op_name))
      if input_tensor_info.op_name not in graph.topo_order:
        print("In %r: input tensor %r points to an op (%r) that does not exist in graph.topo_order" % (op_name, input_tensor_info.name, input_tensor_info.op_name))
        conflicts.append((input_tensor_info.name, input_tensor_info.op_name))


class CMSIS_NN_Transformer(Transformer):
  METHOD_NAME = 'cmsisnn'
  KWARGS_NAMESCOPE = '_utensor_cmsisnn'

  def make_rand_const(self, shape, name):
    val = np.random.random(shape)
    return tf.convert_to_tensor(val, name=name, dtype=tf.float32)

  def get_matcher_graph(self):
    graph = tf.Graph()
    tf.reset_default_graph()
    with graph.as_default():
    
      x = tf.placeholder(dtype=tf.float32, name='input')
      #x = self.make_rand_const([1,784], name='input')
      W_fc1 = self.make_rand_const([784, 128], name='weight')
      b_fc1 = self.make_rand_const([128], name='bias')
      matmal = tf.matmul(x, W_fc1, name='matmal')
      a_fc1 = tf.add(matmal, b_fc1, name="zscore")

      meta = dict()
      #meta["input"] = "Any"
      meta["zscore_eightbit/Variable_1__port__0/quantize"] = ["End", "Any"]
      meta["MatMul_eightbit/x__port__0/quantize"] = ["End", "Any"]
      meta["matmal_eightbit/input/quantize"] = ["End", "Any"]
      meta["zscore_eightbit/bias/quantize"] = ["End", "Any"]

      #import pdb; pdb.set_trace()
    #return (uTensorGraph(graph.as_graph_def(), ['zscore']), meta)
    #ugraph = uTensorGraph(graph.as_graph_def(), ['zscore'])
    #graph_def = ugraph.graph_def
      quant_graph_def = TransformGraph(input_graph_def=graph.as_graph_def(),
                                     inputs=['input`'],
                                     outputs=['zscore'],
                                     transforms=["quantize_weights", "quantize_nodes"])
      mgraph = uTensorGraph(graph=quant_graph_def, output_nodes=['zscore/eightbit/requantize'])
#      import pdb; pdb.set_trace()
#      mgraph = replace_tensors_op("input", new_op_name, ugraph)
#      mgraph.drop_op("input")
#      mgraph.drop_op("matmal_eightbit/input/reshape_dims")
#      mgraph.drop_op("matmal_eightbit/input/reshape")
#      mgraph.drop_op("matmal_eightbit/input/reduction_dims")
#      mgraph.drop_op("matmal_eightbit/input/min")
#      mgraph.drop_op("matmal_eightbit/input/max")
#      #mgraph.drop_op("matmal_eightbit/input/quantize")
#      mgraph.drop_op("weight_quantized_const")
#      mgraph.drop_op("weight_quantized_min")
#      mgraph.drop_op("weight_quantized_max")
    mgraph.viz_graph(fname="matcher.gv")
    return (mgraph, meta)

  def transform(self, ugraph):
    [matcher_ugraph, metaData] = self.get_matcher_graph()
    ugraph.viz_graph(fname="subject.gv")

    while True:
      #import pdb; pdb.set_trace()
      result = isomorphic_match(ugraph, matcher_ugraph, metaData)
      if result == False:
        break
      
      #generate new op name
      #new_op_name = result[0]["zscore/eightbit/requantize"]
      new_op_name = "cmsis_fc_" + result[0]["zscore/eightbit/requantize"]
      #new_op_name = "cmsis_fc_" + result[0]["zscore"]

      #compile new op's the input list
      in_tensors = list()
      #in_tensors.append(tensorInfo_from_name(ugraph, result[1]['weight:0']))
#      import pdb; pdb.set_trace()
      #in_tensors.append(tensorInfo_from_name(ugraph, result[1]['weight_quantized_const:0']))
      #in_tensors.append(tensorInfo_from_name(ugraph, result[1]['input:0']))
      #in_tensors.append(tensorInfo_from_name(ugraph, result[1]['bias:0']))
      in_tensors.append(tensorInfo_from_name(ugraph, result[1]['matmal_eightbit/input/quantize:0']))
      in_tensors.append(tensorInfo_from_name(ugraph, result[1]['matmal_eightbit/input/quantize:1']))
      in_tensors.append(tensorInfo_from_name(ugraph, result[1]['matmal_eightbit/input/quantize:2']))
      in_tensors.append(tensorInfo_from_name(ugraph, result[1]['weight_quantized_const:0']))
      in_tensors.append(tensorInfo_from_name(ugraph, result[1]['weight_quantized_min:0']))
      in_tensors.append(tensorInfo_from_name(ugraph, result[1]['weight_quantized_max:0']))
      in_tensors.append(tensorInfo_from_name(ugraph, result[1]['zscore_eightbit/bias/quantize:0']))
      in_tensors.append(tensorInfo_from_name(ugraph, result[1]['zscore_eightbit/bias/quantize:1']))
      in_tensors.append(tensorInfo_from_name(ugraph, result[1]['zscore_eightbit/bias/quantize:2']))


      # NOTE: Up to here should be correct
      #compile new op's output list
      out_tensors = ugraph.ops_info[result[0]["zscore/eightbit/requantize"]].output_tensors
      #update updating all relevant tensors to point to the new op
      ugraph = replace_tensors_op(result[0]["zscore/eightbit/requantize"], new_op_name, ugraph)

      #FIXME: shouldn't be Tensorflow backend
      tmp_ugraph = uTensorGraph()
      fused_op_info = OperationInfo(name=new_op_name,
                              input_tensors=in_tensors,
                              output_tensors=out_tensors,
                              op_type="CMSIS_NN_FC",
                              backend="tensorflow",
                              ugraph=tmp_ugraph
                              )

      ugraph.drop_op(result[0]['matmal/eightbit'])
      ugraph.drop_op(result[0]['matmal/eightbit/requant_range'])
      ugraph.drop_op(result[0]['matmal/eightbit/requantize'])
      ugraph.drop_op(result[0]['zscore/eightbit'])
      ugraph.drop_op(result[0]['zscore/eightbit/requant_range'])
      ugraph.drop_op(result[0]['zscore/eightbit/requantize'])
      ugraph.add_op(fused_op_info)
      graph_validate(ugraph)

    graph_check(ugraph)
    #import pdb; pdb.set_trace()
    ugraph.viz_graph(fname="cmsis_nn.gv")
    return ugraph
