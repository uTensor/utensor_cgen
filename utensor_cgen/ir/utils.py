# -*- coding: utf8 -*-
from collections import defaultdict
from copy import deepcopy

import tensorflow as tf


def clusters_by_name_scopes(op_infos, name_scope_prefix=None):
  """
  Arguements
  ----------
  op_infos : list[OperationInfo]
      list of parser.OperationInfo
  name_scope_prefix : str
      the target name scope prefix, e.g `dropout`.

  Return
  ------
  clusters : dict
      a dictionary of found name_scopes as key and list of
      operation names as value
  """
  op_infos = deepcopy(op_infos)
  if name_scope_prefix is not None:
    op_infos = [op_info for op_info in op_infos
                if op_info.node_name.startswith(name_scope_prefix)]

  name_scope_map = defaultdict(lambda: [])
  visited = set([])
  for op_info in op_infos:
    current_name_scope = op_infos[0].node_name.split('/')[0]
    if op_info.node_name in visited:
      continue
    queue = [parse_tensor_name(tensor[0])[0]
             for tensor in op_info.input_tensor + op_info.output_tensor]
    cluster = set([])
    while len(queue) > 0:
      op_name = queue.pop(0)
      cluster.add(op_name)
      input_tensors = op_info[op_name].input_tensor
      output_tensors = op_info[op_name].output_tensor
      all_ops = [parse_tensor_name(tensor[0])[0]
                 for tensor in input_tensors + output_tensors]
      queue.extend(all_ops)
    name_scope_map[current_name_scope] = cluster
    visited.update(cluster)
  return name_scope_map


def is_list_of(vtype):
  def check(inst, attrib, value):
    is_valid = True
    if not isinstance(value, list):
      is_valid = False
    else:
      for v in value:
        if not isinstance(v, vtype):
          is_valid = False
    if not is_valid:
      raise TypeError('Expecting list of type %s, get %s' % (vtype, value))
  return check

def graph_check(graph):
  for op_name, op_info in graph.ops_info.items():
    for input_tensor_info in op_info.input_tensors:
      assert input_tensor_info.op_name in graph.ops_info, "In %r: input tensor %r points to non-existing op %r" % (op_name, input_tensor_info.name, input_tensor_info.op_name)
      assert input_tensor_info.op_name in graph.topo_order, "In %r: input tensor %r points to an op (%r) that does not exist in graph.topo_order" % (op_name, input_tensor_info.name, input_tensor_info.op_name)

  #assert len(graph.ops_info) == len(graph.topo_order)