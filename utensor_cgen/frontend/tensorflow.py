from __future__ import absolute_import
import os
import six

import tensorflow as tf
import numpy as np
from google.protobuf import text_format

from utensor_cgen.frontend.base import Parser
from utensor_cgen.frontend import FrontendSelector
from utensor_cgen.ir.base import TensorInfo, OperationInfo, uTensorGraph
from utensor_cgen.utils import topologic_order_graph


@FrontendSelector.register(target_exts=['.pb', '.pbtxt'])
class GraphDefParser(Parser):

  @classmethod
  def parse(cls, pb_file, output_nodes=None):
    graph_def = cls._load_graph_def(pb_file)
    if not cls._tf_is_freeze_graph(graph_def):
      raise ValueError('Given graph_def is not freezed')
    if output_nodes is None:
      output_nodes = [node.name for node in graph_def.node]
    
    graph = tf.Graph()
    with graph.as_default():
      tf.import_graph_def(graph_def, name='')

    ugraph = uTensorGraph(output_nodes=output_nodes,
                          backend="tensorflow")
    for node in graph_def.node:
      op = graph.get_operation_by_name(node.name)
      in_tensors = [TensorInfo(name=tensor.name,
                               ugraph=ugraph,
                               op_name=tensor.op.name,
                               dtype=np.dtype(tensor.dtype.as_numpy_dtype),
                               shape=cls._tf_parse_tshape(tensor.shape))
                    for tensor in op.inputs]
      out_tensors = [TensorInfo(name=tensor.name,
                                ugraph=ugraph,
                                op_name=op.name,
                                dtype=np.dtype(tensor.dtype.as_numpy_dtype),
                                shape=cls._tf_parse_tshape(tensor.shape))
                     for tensor in op.outputs]
      op_type = node.op
      op_attr = node.attr
      op_info = OperationInfo(name=node.name,
                              input_tensors=in_tensors,
                              output_tensors=out_tensors,
                              op_type=op_type,
                              backend='tensorflow',
                              op_attr=op_attr,
                              ugraph=ugraph)
      op_info.op_attr['tensorflow__device'] = node.device
      ugraph.ops_info[node.name] = op_info
    topologic_order_graph(ugraph)
    return ugraph

  @staticmethod
  def _load_graph_def(pb_file):
    if isinstance(pb_file, tf.GraphDef):
      return pb_file
    assert isinstance(pb_file, six.string_types)
    graph_def = tf.GraphDef()
    if pb_file[-3:] == ".pb":
      with open(pb_file, 'rb') as fid:
        graph_def.ParseFromString(fid.read())
    elif pb_file[-3:] == ".pbtxt":
      with open(pb_file, 'r') as fid:
        text_format.Parse(fid.read(), graph_def)
    else:
      raise ValueError('unknown file format: %s' % pb_file)
    return graph_def


  @staticmethod
  def _tf_parse_tshape(t_shape):
    try:
      shape = t_shape.as_list()
    except ValueError:
      shape = None
    return shape

  @classmethod
  def _tf_is_freeze_graph(self, graph_def):
    is_frozen = all(node.op not in ['VariableV2'] for node in graph_def.node)
    return is_frozen