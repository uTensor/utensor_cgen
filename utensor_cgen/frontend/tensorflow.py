from __future__ import absolute_import

import os

import numpy as np
import six
import tensorflow.compat.v1 as tf
from google.protobuf import text_format

from utensor_cgen.frontend import FrontendSelector
from utensor_cgen.frontend.base import Parser
from utensor_cgen.ir.base import OperationInfo, TensorInfo, uTensorGraph
from utensor_cgen.legalizer import Legalizer
from utensor_cgen.logger import logger
from utensor_cgen.utils import random_str, topologic_order_graph


@FrontendSelector.register(target_exts=[".pb", ".pbtxt"])
class GraphDefParser(Parser):

  def parse(self, pb_file, output_nodes=None, model_name=None):
    graph_def, graph_name = self._load_graph_def(pb_file)
    if model_name:
      graph_name = model_name
    if not self._is_freeze_graph(graph_def):
      raise ValueError("Given graph_def is not freezed")
    if output_nodes is None:
      output_nodes = [node.name for node in graph_def.node]
      logger.warning(
        'output_nodes is not given, use all nodes instead (may cause unexpected behaviour)'
      )

    graph = tf.Graph()
    with graph.as_default():
      tf.import_graph_def(graph_def, name="")
    ugraph = uTensorGraph(
      name=graph_name, output_nodes=output_nodes, lib_name="tensorflow",
    )
    for node in graph_def.node:
      op = graph.get_operation_by_name(node.name)
      in_tensors = [
        TensorInfo(
          name=tensor.name,
          ugraph=ugraph,
          op_name=tensor.op.name,
          dtype=np.dtype(tensor.dtype.as_numpy_dtype),
          shape=self._tf_parse_tshape(tensor.shape),
        )
        for tensor in op.inputs
      ]
      out_tensors = [
        TensorInfo(
          name=tensor.name,
          ugraph=ugraph,
          op_name=op.name,
          dtype=np.dtype(tensor.dtype.as_numpy_dtype),
          shape=self._tf_parse_tshape(tensor.shape),
        )
        for tensor in op.outputs
      ]
      op_type = node.op
      op_attr = node.attr
      op_info = OperationInfo(
        name=node.name,
        input_tensors=in_tensors,
        n_inputs=len(in_tensors),
        output_tensors=out_tensors,
        n_outputs=len(out_tensors),
        op_type=op_type,
        lib_name="tensorflow",
        op_attr=op_attr,
        ugraph=ugraph,
      )
      op_info.op_attr["tensorflow__device"] = node.device
      ugraph.ops_info[node.name] = op_info
    topologic_order_graph(ugraph)
    ugraph = Legalizer.legalize(ugraph, {})
    return ugraph

  @staticmethod
  def _load_graph_def(pb_file):
    if isinstance(pb_file, tf.GraphDef):
      return pb_file, "tf_graph_{}".format(random_str(6))
    assert isinstance(pb_file, six.string_types)
    graph_name, ext = os.path.splitext(os.path.basename(pb_file))
    graph_def = tf.GraphDef()
    if ext == ".pb":
      with open(pb_file, "rb") as fid:
        graph_def.ParseFromString(fid.read())
    elif ext == ".pbtxt":
      with open(pb_file, "r") as fid:
        text_format.Parse(fid.read(), graph_def)
    else:
      raise ValueError("unknown file format: %s" % pb_file)
    return graph_def, graph_name

  @staticmethod
  def _tf_parse_tshape(t_shape):
    try:
      shape = t_shape.as_list()
    except ValueError:
      shape = None
    return shape

  @staticmethod
  def _is_freeze_graph(graph_def):
    is_frozen = all(node.op not in ["VariableV2"] for node in graph_def.node)
    return is_frozen
