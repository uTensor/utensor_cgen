# -*- coding:utf8 -*-
import numpy as np
from tensorflow import GraphDef


def _parse_tensor_name(tname: str) -> tuple:
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


def _canonical_input_name(input_name: str) -> str:
  op_name, idx = _parse_tensor_name(input_name)
  return "{}:{}".format(op_name, idx)


def _graph_def_to_map(graph_def: GraphDef) -> dict:
  graph_d = dict((node.name, {"input": set(map(_canonical_input_name, node.input))})
                for node in graph_def.node)
  return graph_d


def _map_to_adjacent(graph_d: dict) -> np.ndarray:
  N = len(graph_d.keys())
  idx_map = dict((node.name, i) for i, node in enumerate(graph_d.keys()))
  adj_mat = np.zeros((N, N))
  for node_name, in_nodes in graph_d.items():
    row_idx = idx_map[node_name]
    for in_node in in_nodes:
      col_idx = idx_map[in_node]
      adj_mat[row_idx, col_idx] = 1.0

  return adj_mat
