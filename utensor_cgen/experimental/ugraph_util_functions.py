import re
from collections import defaultdict
from copy import deepcopy

from utensor_cgen.ir import OperationInfo, uTensorGraph, TensorInfo
from utensor_cgen.ir.converter import AttrValueConverter, GenericTensorConverterMixin # hue hue hue hue hue
from utensor_cgen.utils import parse_tensor_name
from utensor_cgen.ir.utils import graph_check

__all__ = ["is_connected", "get_input_tensor_names", "get_output_tensor_names", "tensorInfo_from_name",
       "get_tensor_node_names", "replace_tensors_op", "replace_tensor_op_by_name",
       "graph_validate", "get_input_node_names", "get_output_node_names", "replace_tensor"]


def is_connected(graph, node0, node1):
  input_nodes = get_input_node_names(graph, node0)
  output_nodes = get_output_node_names(graph, node0)
  node_list = input_nodes.union(output_nodes)

  return node1 in node_list

def get_input_tensor_names(graph, node_name):
  return [t.name for t in graph.ops_info[node_name].input_tensors]

def get_output_tensor_names(graph, node_name):
  return [t.name for t in graph.ops_info[node_name].output_tensors]



def tensorInfo_from_name(graph, edge_name, assertive=True):
  for op_name, op_info in graph.ops_info.items():
    for t in op_info.input_tensors:
      if t.name == edge_name:
        return t
    for t in op_info.output_tensors:
      if t.name == edge_name:
        return t
  assert not assertive, "tensor not %s found" % edge_name
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

def get_input_node_names(graph, node_name):
  input_op_infos = graph.ops_info[node_name].input_nodes
  input_op_names = [op.name for op in input_op_infos]

  return input_op_names

def get_output_node_names(graph, node_name):
  output_op_infos = graph.ops_info[node_name].output_nodes
  output_op_names = [op.name for op in output_op_infos]

  return output_op_names

def replace_tensor(name, new_tensorInfo, ugraph):
  for key, op_info in ugraph.ops_info.items():
    #inputs
    for i, t_info in enumerate(op_info.input_tensors):
      if(t_info.name == name):
        op_info.input_tensors[i] = new_tensorInfo
    #outputs
    for i, t_info in enumerate(op_info.output_tensors):
      if(t_info.name == name):
        op_info.output_tensors[i] = new_tensorInfo