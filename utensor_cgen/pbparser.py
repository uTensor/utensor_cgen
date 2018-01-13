# -*- coding:utf8 -*-
"""
Parser for Protobuf file of Tensorflow Graph
"""
import io
import sys
from collections import namedtuple

import tensorflow as tf
from tensorflow.python.framework import graph_util  # pylint: disable=E0611

from ._pbparser_impl import _parse_graph_def

__all__ = ["parse_pb", "GraphDefParser"]

class GraphDefParser:

  GraphInfo = namedtuple(
    "GraphInfo",
    field_names=["graph_info", "ops_bfs", "output_nodes"])

  @classmethod
  def parse(cls, graph_def, output_nodes=None):
    ops_info, ops_bfs = _parse_graph_def(graph_def, output_nodes)
    return cls.GraphInfo(ops_info, ops_bfs, output_nodes)


def parse_pb(file_or_path, output_nodes=None):
  """
  Arguments
  =========
  - file_or_path: a file object or a path string of the pb file
  - output_nodes: list of output node names

  Returns
  =======
  - graph_info <defaultdict>: a dict with information neccessary for
    building context in uTensor
  - layers <list>: list of layer which is a list of operation names
    in the graph

  Note
  ====
  graph_info example:
    { 'my_const': {
        "input_tensor": set([]),
        "output_tensor": set(["my_const:0"])
        "output_content": {"my_const:0": 3.14},
        "op_type": "Const"
      },
      'myop': {
        "input_tensor": set(["input1:0", "input2:0"]),
        "output_tensor": set(["my_op:0", "my_op:1"]),
        "output_content": {},
        "op_type": "MyOp"
      },
      ...
    }

  layers example:
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
  if sys.version_info.major < 3:
    file_type = (file, io.IOBase)  # pylint: disable=E0602
  else:
    file_type = io.IOBase

  if isinstance(file_or_path, str):
    fid = open(file_or_path, "rb")
  elif isinstance(file_or_path, file_type):
    fid = file_or_path
  else:
    raise ValueError("`file_or_path` has to be either file object or path string")

  # load pb file
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(fid.read())
  fid.close()

  if output_nodes is not None:
    graph_def = graph_util.extract_sub_graph(graph_def, output_nodes)

  ops_info, ops_bfs, output_nodes = _parse_graph_def(graph_def, output_nodes)
  return ops_info, ops_bfs, output_nodes
