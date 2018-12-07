import re
from collections import defaultdict
from copy import deepcopy

from utensor_cgen.ir import OperationInfo, uTensorGraph, TensorInfo
from utensor_cgen.ir.converter import AttrValueConverter, GenericTensorConverterMixin # hue hue hue hue hue
from utensor_cgen.utils import parse_tensor_name
from tensorflow.tools.graph_transforms import TransformGraph
from utensor_cgen.ir.utils import graph_check

from utensor_cgen.experimental.ugraph_util_functions import *

__all__ = ["uGraphMatcher"]

class uGraphMatcher(object):

  translator = None
  subject_graph = None
  matcher_graph = None

  def get_ops_io_info(self, op_type):
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

  def subgraph_trace_exposed_edges(self, graph, start_index=0, end_index=None):
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


  def subgraph_trace_internal_edges(self, graph, start_index=0, end_index=None):
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

  #named sequence
  def forward_path_tracer(self, graph, start_node_name, end_node_name, depth=-1):
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
      forward_path_list = self.forward_path_tracer(graph, output_node_name, end_node_name, depth-1)
      if forward_path_list == None:
        continue
      for forward_path in forward_path_list:
        forward_path.insert(0, start_node_name) #list
        path_list.append(forward_path) #list of list

    if len(path_list) == 0:
      return None

    return path_list
  


  #recursive helper
  #returns False when mismatch
  #returns node-name-relation if a match exist
  def isomorphic_associativity_helper(self, subject_node_name, matcher_node_name, subject_graph, matcher_graph, matcher_meta=None, depth=0, subject_trace=None, matcher_trace=None):
    #compare the current nodes
    subject_node = subject_graph.ops_info[subject_node_name]
    matcher_node = matcher_graph.ops_info[matcher_node_name]
    
    if subject_node.op_type != matcher_node.op_type and not "Any" in self.get_node_meta(matcher_node_name, matcher_meta) :
      return False

    #create and concate the named-relations
    matcher_to_subject_nodes = dict()
    matcher_to_subject_edges = dict()
    matcher_to_subject_nodes[matcher_node_name] = subject_node_name

    matcher_node = matcher_graph.ops_info[matcher_node_name]
    [input_groups, _] = self.get_ops_io_info(matcher_node.op_type)
    #dealing with end of terminations

    if depth <= 0 or len(matcher_node.output_tensors) == 0 or input_groups == None or "End" in self.get_node_meta(matcher_node_name, matcher_meta):
      if subject_trace != None and matcher_trace != None:
        subject_start_node_name = subject_trace[0]
        subject_path_group = self.forward_path_tracer(subject_graph, subject_node_name, subject_start_node_name, len(matcher_graph.topo_order)-1)

        matcher_start_node_name = matcher_trace[0]
        matcher_path_group = self.forward_path_tracer(matcher_graph, matcher_node_name, matcher_start_node_name, len(matcher_graph.topo_order)-1)

        if not self.compare_paths(subject_path_group, matcher_path_group, subject_graph, matcher_graph, None, matcher_meta):
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

          probe = self.isomorphic_associativity_helper(sweeping_subject_input_node_name[0], matcher_input_node_name[0], subject_graph, matcher_graph, matcher_meta, depth-1, subject_trace, matcher_trace)
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

  def isomorphic_match(self, subject_graph, matcher_graph, meta):
    matcher_to_subject_nodes = dict()
    matcher_to_subject_edges = dict()
    matcher_output_node_names = set()

    self.subject_graph = subject_graph
    self.matcher_graph = matcher_graph

    #identify matcher output nodes
    [_, matcher_output_edges] = self.subgraph_trace_exposed_edges(matcher_graph)
    
    for matcher_output_edge in matcher_output_edges: 
      [tensor_origin, _] = get_tensor_node_names(matcher_graph, matcher_output_edge)
      matcher_output_node_names.add(tensor_origin[0])
    max_search_depth = len(matcher_graph.topo_order)
    partial_matcher_to_subject_nodes = None
    partial_matcher_to_subject_edges = None
    #support only signle matcher output node for now
    for subject_node_name in subject_graph.topo_order:
      probe = self.isomorphic_associativity_helper(subject_node_name, next(iter(matcher_output_node_names)), subject_graph, matcher_graph, meta, max_search_depth)
      if probe == False:
        continue
      [partial_matcher_to_subject_nodes, partial_matcher_to_subject_edges] = probe

    if partial_matcher_to_subject_nodes == None and partial_matcher_to_subject_edges == None:
      return False

    matcher_to_subject_nodes.update(partial_matcher_to_subject_nodes)
    matcher_to_subject_edges.update(partial_matcher_to_subject_edges)

    self.translator = [matcher_to_subject_nodes, matcher_to_subject_edges]

    return self.translator

  def get_node_meta(self, node_name, meta):
    if meta == None:
      return ["None"]
    if not node_name in meta:
      return ["None"]
    return meta[node_name]

  #TODO: FIXME:
  #this function only has heurstic comparator
  def compare_paths(self, path_group0, path_group1, graph0, graph1, meta0=None, meta1=None):
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

          if node0.op_type != node1.op_type and (not "Any" in self.get_node_meta(node0_name, meta0) and not "Any" in self.get_node_meta(node1_name, meta1)):
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

  def tensor_info(self, name):
    return tensorInfo_from_name(self.subject_graph, self.translator[1][name])
  
  def op_info(self, name):
    return self.subject_graph.ops_info[self.translator[0][name]]

  def __getitem__(self, name):
    if name in self.translator[0]:
      return self.op_info(name)
    if name in self.translator[1]:
      return self.tensor_info(name)
    
    assert "% not found\r\n", name

  def __setitem__(self, name, info):
    if isinstance(info, TensorInfo):
      replace_tensor(self.translator[1][name], info, self.subject_graph)
      self.translator[1][name] = info.name
      return

    if isinstance(info, OperationInfo):
      replace_tensors_op(self.translator[0][name], info.name, self.subject_graph)
      self.subject_graph.ops_info[self.translator[0][name]] = info
      self.translator[0][name] = info.name
      return
    
    if info == None:
      #TODO: tensor dependency checking here
      if name in self.subject_graph.topo_order:
        self.subject_graph.drop_op(self.translator[0][name])
        del self.translator[0][name]
      else:
        assert "% not found\r\n", name
      return
    
    assert False