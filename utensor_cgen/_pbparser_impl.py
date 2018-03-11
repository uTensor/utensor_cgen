# -*- coding:utf8 -*-
from collections import defaultdict, namedtuple

import numpy as np
from tensorflow import Graph, Session, import_graph_def


def _parse_tensor_name(tname):
  """Adapt from TensorFlow source code
  """
  components = tname.split(":")
  if len(components) == 2:
    try:
      output_index = int(components[1])
    except ValueError:
      raise ValueError("invalid output index: {}".format(tname))
    return (components[0], output_index)
  elif len(components) == 1:
    return (components[0], 0)
  else:
    raise ValueError("invalid tensor name: {}".format(tname))


def _op_name(input_name):
  return _parse_tensor_name(input_name)[0]


def _graph_def_to_map(graph_def):
  """Return a mapping from operation name to a set
  of input operation names
  """
  graph_d = dict((node.name, set(map(_op_name, node.input)))
                 for node in graph_def.node)
  return graph_d


def _map_to_adjacent(graph_d):
  N = len(graph_d.keys())
  idx_map = dict((name, i) for i, name in enumerate(graph_d.keys()))
  inv_idx_map = dict((idx, name) for name, idx in idx_map.items())
  adj_mat = np.zeros((N, N))
  for node_name, in_nodes in graph_d.items():
    row_idx = idx_map[node_name]
    for in_node in in_nodes:
      col_idx = idx_map[in_node]
      adj_mat[row_idx, col_idx] = 1.0

  return adj_mat, inv_idx_map


def _graph_def_to_adjacent(graph_def):
  graph_d = _graph_def_to_map(graph_def)
  return _map_to_adjacent(graph_d)


def _has_content(node):
  if 'value' in node.attr:
    return False
  value = node.attr['value']
  if not hasattr(value, 'tensor'):
    return False
  return True


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
  OperationInfo = namedtuple('OperationInfo',
                             field_names=['input_tensor', 'output_tensor',
                                          'op_type', 'output_content'])
  graph = Graph()
  with graph.as_default():  # pylint: disable=E1129
    import_graph_def(graph_def, name="")
  graph_info = {}
  with Session(graph=graph):
    for node in graph_def.node:
      op = graph.get_operation_by_name(node.name)
      input_tensor = [(t.name, t.dtype, _parse_shape(t.shape)) for t in op.inputs]
      output_tensor = [(t.name, t.dtype, _parse_shape(t.shape)) for t in op.outputs]
      op_type = node.op
      output_content = {}
      if node.op in ["Const"]:
        for tensor_name, _, _ in output_tensor:
          tensor = graph.get_tensor_by_name(tensor_name)
          output_content[tensor_name] = tensor.eval()
      graph_info[node.name] = OperationInfo(input_tensor, output_tensor, op_type, output_content)
  return graph_info


def _parse_graph_layers(graph_def):
  """Devide graph into layers (Fast way to find output nodes)

  Argument
  ========
  - graph_def <GraphDef>: protobuf GraphDef object

  Return
  ======
  - layers: list of layer which is a list of operation name

  Note
  ====
  Ex:
    `bottom` <--------> `top`
      foo -
            \\
              tar - - var
            /
      bar -
  the return list, layers, will be [['foo', 'bar'], ['tar'], ['var']]
  That is, layers[0] is the bottom layer of the graph, layers[1] is the
  second bottom layer of the graph, so on and so forth
  """
  adj_mat, inv_idx_map = _graph_def_to_adjacent(graph_def)
  state_v = np.ones(adj_mat.shape[0])
  layers = []
  visited = set([])
  while (state_v > 0).sum() != 0:
    state_v = adj_mat.dot(state_v)
    node_idx = set(np.where(state_v == 0)[0])
    layer = [inv_idx_map[i] for i in node_idx - visited]
    layers.append(layer)
    visited.update(node_idx)
  return layers


def _is_freeze_graph(graph_def):
  is_frozen = all(node.op not in ['VariableV2'] for node in graph_def.node)
  return is_frozen


def _parse_graph_topologic_order(graph_def, output_nodes=None):
  # https://en.wikipedia.org/wiki/Topological_sorting
  if output_nodes is None:
    output_nodes = _parse_graph_layers(graph_def)[-1]
  graph = Graph()
  with graph.as_default():
    import_graph_def(graph_def, name='')

  queue = output_nodes.copy()
  visited = set()    # temporary mark
  perm_visit = set() # Permanent mark
  ops_bfs = [] # L

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
    ops_bfs.insert(0, node_name)

  while queue:
    node_name = queue.pop(0)
    visit(node_name)    

  # ops_bfs.reverse()
  return ops_bfs, output_nodes


def _parse_graph_def(graph_def, output_nodes=None):
  if not _is_freeze_graph(graph_def):
    raise ValueError("The graph is not frozen, freeze the graph first")
  ops_bfs, output_nodes = _parse_graph_topologic_order(graph_def, output_nodes=output_nodes)
  ops_info = _parse_graph_info(graph_def)
  return ops_info, ops_bfs, output_nodes
