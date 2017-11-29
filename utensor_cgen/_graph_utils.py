# -*- coding:utf8 -*-
import numpy as np
from tensorflow import GraphDef


def _parse_tensor_name(tname: str) -> (str, int):
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


def _op_name(input_name: str) -> str:
  return _parse_tensor_name(input_name)[0]


def _graph_def_to_map(graph_def: GraphDef) -> dict:
  """Return a mapping from operation name to a set
  of input operation names
  """
  graph_d = dict((node.name, set(map(_op_name, node.input)))
                 for node in graph_def.node)
  return graph_d


def _map_to_adjacent(graph_d: dict) -> (np.ndarray, dict):
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


def _graph_def_to_adjacent(graph_def: GraphDef) -> (np.ndarray, dict):
  graph_d = _graph_def_to_map(graph_def)
  return _map_to_adjacent(graph_d)


def _parse_graph_layers(graph_def: GraphDef) -> list:
  adj_mat, inv_idx_map = _graph_def_to_adjacent(graph_def)
  state_v = np.ones(adj_mat.shape[0])
  layers = []
  visited = set([])
  while (state_v > 0).sum() != 0:
    state_v = adj_mat @ state_v
    node_idx = set(np.where(state_v == 0)[0])
    layer = [inv_idx_map[i] for i in node_idx - visited]
    layers.append(layer)
    visited.update(node_idx)
  return layers


def _parse_graph_nodes(graph_def: GraphDef) -> dict:
  graph_nodes = dict((node.name, {"input": set(node.input), "output": set([])})
                     for node in graph_def.node)
  for op_name in graph_nodes.keys():
    op_info = graph_nodes[op_name]
    input_op_names = map(_op_name, op_info["input"])
    for in_op in input_op_names:
      graph_nodes[in_op]["output"].add(op_name)
  return graph_nodes


def _parse_graph_def(graph_def: GraphDef) -> (dict, list):
  layers = _parse_graph_layers(graph_def)
  graph_nodes = _parse_graph_nodes(graph_def)
  return graph_nodes, layers
