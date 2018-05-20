# -*- coding:utf8 -*-
from collections import defaultdict, namedtuple
from copy import deepcopy

import numpy as np
from tensorflow import Graph, Session, import_graph_def
from tensorflow.contrib.util import make_ndarray
from utensor_cgen.parser.types import OperationInfo, TensorInfo


def _parse_shape(t_shape):
  try:
    shape = t_shape.as_list()
  except ValueError:
    shape = None
  return shape


def _parse_graph_info(graph_def):
  """Parse GraphDef
  Fetch input tensors and output tensors name for reconstructing
  graph in uTensor Context object

  Argument
  ========
  - graph_def <tf.GraphDef>: a GraphDef object

  Return
  ======
  - graph_nodes <defaultdict>: a dict with key as operation name and
    value as a defaultdict with keys 'input_tensor' and 'output_tensor'
    which maps to a set of input/output tensor names respectively

  Note
  ====
  - thought the output tensor names is irrelevent for TensorFlow, but it
    is neccessary for uTensor
  """
  graph = Graph()
  with graph.as_default():  # pylint: disable=E1129
    import_graph_def(graph_def, name="")
  graph_info = {}
  with Session(graph=graph):
    for node in graph_def.node:
      op = graph.get_operation_by_name(node.name)
      input_tensor = [TensorInfo(t.name, t.dtype, _parse_shape(t.shape)) for t in op.inputs]
      output_tensor = [TensorInfo(t.name, t.dtype, _parse_shape(t.shape)) for t in op.outputs]
      op_type = node.op
      output_content = {}
      op_attr = node.attr
      if node.op in ["Const"]:
        for tensor_info in output_tensor:
          tensor_name = tensor_info.name
          output_content[tensor_name] = make_ndarray(node.attr['value'].tensor)
      graph_info[node.name] = OperationInfo(node_name=node.name,
                                            input_tensor=input_tensor,
                                            output_tensor=output_tensor,
                                            op_type=op_type,
                                            output_content=output_content,
                                            op_attr=op_attr)
  return graph_info


def _is_freeze_graph(graph_def):
  is_frozen = all(node.op not in ['VariableV2'] for node in graph_def.node)
  return is_frozen


def _parse_graph_topologic_order(graph_def, output_nodes):
  # https://en.wikipedia.org/wiki/Topological_sorting
  graph = Graph()
  with graph.as_default():
    import_graph_def(graph_def, name='')

  queue = deepcopy(output_nodes)
  visited = set()    # temporary mark
  perm_visit = set()  # Permanent mark
  ops_torder = []  # L

  def visit(node_name):
    if node_name in perm_visit:
      return
    if node_name in visited:
      raise ValueError("Input graph is not a DAG")

    visited.add(node_name)
    op = graph.get_operation_by_name(node_name)

    for tensor in op.inputs:
      visit(tensor.op.name)

    perm_visit.add(node_name)
    ops_torder.insert(0, node_name)

  while queue:
    node_name = queue.pop(0)
    visit(node_name)
  return ops_torder


def _parse_graph_def(graph_def, output_nodes):
  if not _is_freeze_graph(graph_def):
    raise ValueError("The graph is not frozen, freeze the graph first")
  ops_topo = _parse_graph_topologic_order(graph_def=graph_def,
                                          output_nodes=output_nodes)
  ops_info = _parse_graph_info(graph_def)
  return ops_info, ops_topo
