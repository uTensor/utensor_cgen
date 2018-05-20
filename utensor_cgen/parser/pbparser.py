# -*- coding:utf8 -*-
"""
Parser for Protobuf file of Tensorflow Graph
"""
import io
import sys

import tensorflow as tf
from tensorflow.python.framework import graph_util  # pylint: disable=E0611

from ._pbparser_impl import _parse_graph_def

__all__ = ["parse_pb"]


def parse_pb(file_or_path, output_nodes):
  """
  Arguments
  =========
  - file_or_path: a file object or a path string of the pb file
  - output_nodes: list of output node names

  Returns
  =======
  - ops_info <dict>: a dict with information neccessary for
    building context in uTensor
  - ops_topo <list>: list of op node names in topological sorted order
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

  sub_graph_def = graph_util.extract_sub_graph(graph_def,
                                               output_nodes)

  ops_info, ops_topo = _parse_graph_def(sub_graph_def, output_nodes)
  return ops_info, ops_topo
