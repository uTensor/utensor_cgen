# -*- coding:utf8 -*-
r"""CMSIS-NN Transformer

Node fusion and replacement for CMSIS-NN

"""
import re
from collections import defaultdict
from copy import deepcopy

import tensorflow as tf
import numpy as np

from utensor_cgen.ir import OperationInfo, uTensorGraph, TensorInfo
from utensor_cgen.ir.converter import AttrValueConverter, GenericTensorConverterMixin # hue hue hue hue hue
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

def replace_tensor_op_by_name(tensor_name, new_node_name, graph):
  for op_name, op_info in graph.ops_info.items():
    for i, output_tensor_info in enumerate(op_info.output_tensors):
      if output_tensor_info.name == tensor_name:
        output_tensor_info.op_name = new_node_name
        op_info.output_tensors[i] = output_tensor_info
        graph.ops_info[op_name] = op_info

    for i, input_tensor_info in enumerate(op_info.input_tensors):
      if input_tensor_info.name == tensor_name:
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

# Let us get unique names for custom injected nodes
def static_vars(**kwargs):
  def decorate(func):
    for k in kwargs:
      setattr(func, k, kwargs[k])
    return func
  return decorate

@static_vars(counters=defaultdict(int))
def get_unique_number(target):
  get_unique_number.counters[target] += 1
  return get_unique_number.counters[target]

def bs_ops_attr(np_array):
  """ Simplify creating the op_attr bullshit boilerplate from an numpy array """
  op_attr = {}
  op_attr["tensorflow_device"] = ''
  op_attr["dtype"] = AttrValueConverter.GenericType(value_name='type', value=3) # The hell are these values?
  op_attr["value"] = AttrValueConverter.GenericType(value_name='tensor', 
                                          value=GenericTensorConverterMixin.GenericType(np_array=np_array, 
                                                                                        dtype=np_array.dtype
                                                                                        )
                                         )
  return op_attr

## MatMul Only
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
      meta["matmal_eightbit/input/quantize"] = ["End", "Any"]

      quant_graph_def = TransformGraph(input_graph_def=graph.as_graph_def(),
                                     inputs=['input`'],
                                     outputs=['zscore'],
                                     transforms=["quantize_weights", "quantize_nodes"])
      mgraph = uTensorGraph(graph=quant_graph_def, output_nodes=['matmal/eightbit'])

    #mgraph.viz_graph(fname="matcher.gv")
    return (mgraph, meta)

  def transform(self, ugraph):
    [matcher_ugraph, metaData] = self.get_matcher_graph()
    #ugraph.viz_graph(fname="subject.gv")

    while True:
      result = isomorphic_match(ugraph, matcher_ugraph, metaData)
      if result == False:
        break
      
      tmp_ugraph = uTensorGraph()

      ## convert the inputs Uint8Q7OriginOp
      new_input0_op_name = "convert_uint8_q7_" + result[0]["matmal_eightbit/input/quantize"]
      new_input1_op_name = "convert_uint8_q7_" + result[0]["weight_quantized_const"]

      input0_shape = tensorInfo_from_name(ugraph, result[1]["matmal_eightbit/input/quantize:0"]).shape
      input0_q7_out = TensorInfo(name=result[0]["matmal_eightbit/input/quantize"] + "_q7:0",
                        op_name=new_input0_op_name,  #ownership
                        dtype=np.dtype('int8'),
                        shape=input0_shape,
                        ugraph=tmp_ugraph
                        )
      input0_q7_inputs = list()
      input0_q7_inputs.append(tensorInfo_from_name(ugraph, result[1]["matmal_eightbit/input/quantize:0"]))
      input0_q7_inputs.append(tensorInfo_from_name(ugraph, result[1]["matmal_eightbit/input/quantize:1"]))
      input0_q7_inputs.append(tensorInfo_from_name(ugraph, result[1]["matmal_eightbit/input/quantize:2"]))
      input0_q7_op_info = OperationInfo(name=new_input0_op_name,
                                input_tensors=input0_q7_inputs,
                                output_tensors=[input0_q7_out],
                                op_type="Uint8Q7OriginOp",
                                backend="tensorflow",
                                ugraph=tmp_ugraph,
                                op_attr=bs_ops_attr(np.array([0], dtype=np.uint16))
                                )
      ugraph.add_op(input0_q7_op_info)

      input1_shape = tensorInfo_from_name(ugraph, result[1]["weight_quantized_const:0"]).shape
      input1_q7_out = TensorInfo(name=result[0]["weight_quantized_const"] + "_q7:0",
                        op_name=new_input1_op_name,  #ownership
                        dtype=np.dtype('int8'),
                        shape=input1_shape,
                        ugraph=tmp_ugraph
                        )
      input1_q7_inputs = list()
      input1_q7_inputs.append(tensorInfo_from_name(ugraph, result[1]["weight_quantized_const:0"]))
      input1_q7_inputs.append(tensorInfo_from_name(ugraph, result[1]["weight_quantized_min:0"]))
      input1_q7_inputs.append(tensorInfo_from_name(ugraph, result[1]["weight_quantized_max:0"]))
      input1_q7_op_info = OperationInfo(name=new_input1_op_name,
                                input_tensors=input1_q7_inputs,
                                output_tensors=[input1_q7_out],
                                op_type="Uint8Q7OriginOp",
                                backend="tensorflow",
                                ugraph=tmp_ugraph,
                                op_attr=bs_ops_attr(np.array([0], dtype=np.uint16))
                                )
      ugraph.add_op(input1_q7_op_info)

      #using CMSIS-NN FC as MatMul only, for now
      #generate new op name
      new_op_name = "cmsis_fc_" + result[0]["matmal/eightbit"]

      #import pdb; pdb.set_trace()
      bShift = "cmsis_bShift_"
      bShiftIter = get_unique_number(bShift)
      bShift += "%d" % bShiftIter
      bShift_tensor = TensorInfo(name=bShift + ":0",
                              op_name=bShift,
                              dtype=np.dtype('uint16'),
                              shape=[1],
                              ugraph=tmp_ugraph
                              )
      bShift_op_info = OperationInfo(name=bShift,
                              input_tensors=list(),
                              output_tensors=[bShift_tensor],
                              op_type="Const",
                              backend="tensorflow",
                              ugraph=tmp_ugraph,
                              op_attr=bs_ops_attr(np.array([0], dtype=np.uint16))
                              )
      ugraph.add_op(bShift_op_info)

      oShift = "cmsis_oShift_"
      oShiftIter = get_unique_number(oShift)
      oShift += "%d" % oShiftIter
      oShift_tensor = TensorInfo(name=oShift + ":0",
                              op_name=oShift,
                              dtype=np.dtype('uint16'),
                              shape=[1],
                              ugraph=tmp_ugraph
                              )
      oShift_op_info = OperationInfo(name=oShift,
                              input_tensors=list(),
                              output_tensors=[oShift_tensor],
                              op_type="Const",
                              backend="tensorflow",
                              ugraph=tmp_ugraph,
                              op_attr=bs_ops_attr(np.array([0], dtype=np.uint16))
                              )
      ugraph.add_op(oShift_op_info)

      scratch_space = "cmsis_scratch_"
      scratchIter = get_unique_number(scratch_space)
      scratch_space += "%d" % scratchIter
      scratch_shape = list(map(lambda x: x if x else 1, tensorInfo_from_name(ugraph, result[1]['matmal_eightbit/input/quantize:0']).shape))
      scratch_tensor = TensorInfo(name=scratch_space + ":0",
                              op_name=scratch_space,
                              dtype=np.dtype('uint16'),
                              shape=scratch_shape,
                              ugraph=tmp_ugraph
                              )
      scratch_op_info = OperationInfo(name=scratch_space,
                              input_tensors=list(),
                              output_tensors=[scratch_tensor],
                              op_type="Const",  #fixme
                              backend="tensorflow",
                              ugraph=tmp_ugraph,
                              op_attr=bs_ops_attr(np.zeros(tuple(scratch_shape), dtype=np.uint16))
                              )
      ugraph.add_op(scratch_op_info)
      #import pdb; pdb.set_trace()
      #compile new op's the input list
      in_tensors = list()
      in_tensors.append(input0_q7_out) #pV
      in_tensors.append(input1_q7_out) # Weights pM
      ##Bias is temporary disabled in CMSIS_NN_FC, so give it anything for now
      in_tensors.append(input0_q7_out) # Bias
      in_tensors.append(bShift_tensor)
      in_tensors.append(oShift_tensor)
      in_tensors.append(scratch_tensor)

      #TODO:
      # make sure weight and bias are in the format
      # supply S_TENSOR bShift and S_TENSOR oShift here
      # This can be found by either offline quantization profiling
      # Or, by taking statistic of the weight matrix
      # In uTensor runtime CMSIS Op, convert the number format back to the linear quantized format
      # update _CMSIS_NN_FCOperator in operator.py

      #compile new op's output list
      new_op_name = "cmsis_fc_" + result[0]["matmal/eightbit"]
      matmal_output_tensors = ugraph.ops_info[result[0]["matmal/eightbit"]].output_tensors
      #update updating all relevant tensors to point to the new op
      ugraph = replace_tensors_op(result[0]["matmal/eightbit"], new_op_name, ugraph)

      #FIXME: shouldn't be Tensorflow backend
      fused_op_info = OperationInfo(name=new_op_name,
                              input_tensors=in_tensors,
                              output_tensors=[matmal_output_tensors[0]],
                              op_type="CMSIS_NN_FC",
                              backend="tensorflow",
                              ugraph=tmp_ugraph
                              )

      ugraph.drop_op(result[0]['matmal/eightbit'])
      # ugraph.drop_op(result[0]['matmal/eightbit/requant_range'])
      # ugraph.drop_op(result[0]['matmal/eightbit/requantize'])
      # ugraph.drop_op(result[0]['zscore/eightbit'])
      # ugraph.drop_op(result[0]['zscore/eightbit/requant_range'])
      # ugraph.drop_op(result[0]['zscore/eightbit/requantize'])
      ugraph.add_op(fused_op_info)

      new_range_op_name = new_op_name + "_range"
      # matmul_min_tensor = TensorInfo(name=new_op_name + ":1",
      #                         op_name=new_range_op_name,
      #                         dtype=np.dtype('float'),
      #                         shape=[1],
      #                         ugraph=tmp_ugraph
      #                         )
      # matmul_max_tensor = TensorInfo(name=new_op_name + ":2",
      #                         op_name=new_range_op_name,
      #                         dtype=np.dtype('float'),
      #                         shape=[1],
      #                         ugraph=tmp_ugraph
      #                         )
      new_range_op_inputs = list()
      new_range_op_inputs.append(tensorInfo_from_name(ugraph, result[1]["matmal_eightbit/input/quantize:1"]))
      new_range_op_inputs.append(tensorInfo_from_name(ugraph, result[1]["matmal_eightbit/input/quantize:2"]))
      new_range_op_inputs.append(tensorInfo_from_name(ugraph, result[1]["weight_quantized_min:0"]))
      new_range_op_inputs.append(tensorInfo_from_name(ugraph, result[1]["weight_quantized_max:0"]))

      new_range_op_outputs = list()
      new_range_op_outputs.append(matmal_output_tensors[1])
      new_range_op_outputs.append(matmal_output_tensors[2])
      ugraph = replace_tensor_op_by_name(matmal_output_tensors[1].name, new_range_op_name, ugraph)
      ugraph = replace_tensor_op_by_name(matmal_output_tensors[2].name, new_range_op_name, ugraph)
      new_range_op_outputs[0].op_name = new_range_op_name
      new_range_op_outputs[1].op_name = new_range_op_name
      new_range_op_info = OperationInfo(name=new_range_op_name,
                        input_tensors=new_range_op_inputs,
                        output_tensors=new_range_op_outputs,
                        op_type="QuantRangeForMultiplicationu8u8int32Op",
                        backend="tensorflow",
                        ugraph=tmp_ugraph
                        )
      ugraph.add_op(new_range_op_info)
      graph_validate(ugraph)

    graph_check(ugraph)
    ugraph.viz_graph(fname="cmsis_nn.gv")
    return ugraph


## MatMul + Add
# class CMSIS_NN_Transformer(Transformer):
#   METHOD_NAME = 'cmsisnn'
#   KWARGS_NAMESCOPE = '_utensor_cmsisnn'

#   def make_rand_const(self, shape, name):
#     val = np.random.random(shape)
#     return tf.convert_to_tensor(val, name=name, dtype=tf.float32)

#   def get_matcher_graph(self):
#     graph = tf.Graph()
#     tf.reset_default_graph()
#     with graph.as_default():
    
#       x = tf.placeholder(dtype=tf.float32, name='input')
#       #x = self.make_rand_const([1,784], name='input')
#       W_fc1 = self.make_rand_const([784, 128], name='weight')
#       b_fc1 = self.make_rand_const([128], name='bias')
#       matmal = tf.matmul(x, W_fc1, name='matmal')
#       a_fc1 = tf.add(matmal, b_fc1, name="zscore")

#       meta = dict()
#       #meta["input"] = "Any"
#       meta["matmal_eightbit/input/quantize"] = ["End", "Any"]
#       meta["zscore_eightbit/bias/quantize"] = ["End", "Any"]

#       quant_graph_def = TransformGraph(input_graph_def=graph.as_graph_def(),
#                                      inputs=['input`'],
#                                      outputs=['zscore'],
#                                      transforms=["quantize_weights", "quantize_nodes"])
#       mgraph = uTensorGraph(graph=quant_graph_def, output_nodes=['zscore/eightbit/requantize'])

#     #mgraph.viz_graph(fname="matcher.gv")
#     return (mgraph, meta)

#   def transform(self, ugraph):
#     [matcher_ugraph, metaData] = self.get_matcher_graph()
#     #ugraph.viz_graph(fname="subject.gv")

#     while True:
#       result = isomorphic_match(ugraph, matcher_ugraph, metaData)
#       if result == False:
#         break
      
#       tmp_ugraph = uTensorGraph()
      
#       #generate new op name
#       new_op_name = "cmsis_fc_" + result[0]["zscore/eightbit/requantize"]

#       #import pdb; pdb.set_trace()
#       bShift = "cmsis_bShift_"
#       bShiftIter = get_unique_number(bShift)
#       bShift += "%d" % bShiftIter
#       bShift_tensor = TensorInfo(name=bShift + ":0",
#                               op_name=bShift,
#                               dtype=np.dtype('uint16'),
#                               shape=[1],
#                               ugraph=tmp_ugraph
#                               )
#       bShift_op_info = OperationInfo(name=bShift,
#                               input_tensors=list(),
#                               output_tensors=[bShift_tensor],
#                               op_type="Const",
#                               backend="tensorflow",
#                               ugraph=tmp_ugraph,
#                               op_attr=bs_ops_attr(np.array([0], dtype=np.uint16))
#                               )
#       ugraph.add_op(bShift_op_info)

#       oShift = "cmsis_oShift_"
#       oShiftIter = get_unique_number(oShift)
#       oShift += "%d" % oShiftIter
#       oShift_tensor = TensorInfo(name=oShift + ":0",
#                               op_name=oShift,
#                               dtype=np.dtype('uint16'),
#                               shape=[1],
#                               ugraph=tmp_ugraph
#                               )
#       oShift_op_info = OperationInfo(name=oShift,
#                               input_tensors=list(),
#                               output_tensors=[oShift_tensor],
#                               op_type="Const",
#                               backend="tensorflow",
#                               ugraph=tmp_ugraph,
#                               op_attr=bs_ops_attr(np.array([0], dtype=np.uint16))
#                               )
#       ugraph.add_op(oShift_op_info)

#       scratch_space = "cmsis_scratch_"
#       scratchIter = get_unique_number(scratch_space)
#       scratch_space += "%d" % scratchIter
#       scratch_shape = list(map(lambda x: x if x else 1, tensorInfo_from_name(ugraph, result[1]['matmal_eightbit/input/quantize:0']).shape))
#       scratch_tensor = TensorInfo(name=scratch_space + ":0",
#                               op_name=scratch_space,
#                               dtype=np.dtype('uint16'),
#                               shape=scratch_shape,
#                               ugraph=tmp_ugraph
#                               )
#       scratch_op_info = OperationInfo(name=scratch_space,
#                               input_tensors=list(),
#                               output_tensors=[scratch_tensor],
#                               op_type="Const",
#                               backend="tensorflow",
#                               ugraph=tmp_ugraph,
#                               op_attr=bs_ops_attr(np.zeros(tuple(scratch_shape), dtype=np.uint16))
#                               )
#       ugraph.add_op(scratch_op_info)
#       #import pdb; pdb.set_trace()
#       #compile new op's the input list
#       in_tensors = list()
#       in_tensors.append(tensorInfo_from_name(ugraph, result[1]['matmal_eightbit/input/quantize:0'])) #pV
#       in_tensors.append(tensorInfo_from_name(ugraph, result[1]['weight_quantized_const:0'])) # Weights pM
#       in_tensors.append(tensorInfo_from_name(ugraph, result[1]['zscore_eightbit/bias/quantize:0'])) # Bias
#       in_tensors.append(bShift_tensor)
#       in_tensors.append(oShift_tensor)
#       in_tensors.append(scratch_tensor)

#       #TODO:
#       # make sure weight and bias are in the format
#       # supply S_TENSOR bShift and S_TENSOR oShift here
#       # This can be found by either offline quantization profiling
#       # Or, by taking statistic of the weight matrix
#       # In uTensor runtime CMSIS Op, convert the number format back to the linear quantized format
#       # update _CMSIS_NN_FCOperator in operator.py

#       #compile new op's output list
#       out_tensors = ugraph.ops_info[result[0]["zscore/eightbit/requantize"]].output_tensors
#       #update updating all relevant tensors to point to the new op
#       ugraph = replace_tensors_op(result[0]["zscore/eightbit/requantize"], new_op_name, ugraph)

#       #FIXME: shouldn't be Tensorflow backend
#       fused_op_info = OperationInfo(name=new_op_name,
#                               input_tensors=in_tensors,
#                               output_tensors=out_tensors,
#                               op_type="CMSIS_NN_FC",
#                               backend="tensorflow",
#                               ugraph=tmp_ugraph
#                               )

#       ugraph.drop_op(result[0]['matmal/eightbit'])
#       ugraph.drop_op(result[0]['matmal/eightbit/requant_range'])
#       ugraph.drop_op(result[0]['matmal/eightbit/requantize'])
#       ugraph.drop_op(result[0]['zscore/eightbit'])
#       ugraph.drop_op(result[0]['zscore/eightbit/requant_range'])
#       ugraph.drop_op(result[0]['zscore/eightbit/requantize'])
#       ugraph.add_op(fused_op_info)
#       graph_validate(ugraph)

#     graph_check(ugraph)
#     #ugraph.viz_graph(fname="cmsis_nn.gv")
#     return ugraph
