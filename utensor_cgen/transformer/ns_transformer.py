# -*- coding:utf8 -*-
r"""Namescope Transformer

Transformers that get rid of namescope/nodes which are not needed 
for inference
"""
import re
from collections import defaultdict
from copy import deepcopy

from utensor_cgen.ir import OperationInfo, uTensorGraph
from utensor_cgen.utils import parse_tensor_name

from .base import Transformer

__all__ = ["DropoutTransformer", "BatchNormTransformer", "InlineTransformer"]


class BiasAddTransformer(Transformer):
  METHOD_NAME = 'biasAdd'
  KWARGS_NAMESCOPE = '_utensor_biasAdd'
  TARGET_NODENAME_PATTERN = re.compile(r'(BiasAdd[_\w\d]*)/.*')

  def transform(self, ugraph):
    for node_name in ugraph.topo_order:
      op_type = ugraph.ops_info[node_name].op_type
      if op_type == 'QuantizedBiasAdd':
        op_info = ugraph.ops_info[node_name]
        op_info.op_type = 'QuantizedAdd'
      elif op_type == 'BiasAdd':
        op_info = ugraph.ops_info[node_name]
        op_info.op_type = 'Add'


    return ugraph


class InlineTransformer(Transformer):
  METHOD_NAME = 'inline'
  KWARGS_NAMESCOPE = '_utensor_inline'
  TARGET_NODENAME_PATTERN = re.compile(r'(const[_\w\d]*)/.*')


  def transform(self, ugraph):
    for node_name in ugraph.topo_order:
      op_type = ugraph.ops_info[node_name].op_type
      if op_type == 'Const':
        op_info = ugraph.ops_info[node_name]
        op_info.op_type = 'Inline'
    
    return ugraph

class DropoutTransformer(Transformer):
  """Remove Dropout Op
  """
  METHOD_NAME = 'dropout'
  KWARGS_NAMESCOPE = '_utensor_dropout'
  TARGET_NODENAME_PATTERN = re.compile(r'(dropout[_\w\d]*)/.*')

  def transform(self, ugraph):
    new_graph = uTensorGraph(output_nodes=ugraph.output_nodes)
    dropout_input_map = self._find_input(ugraph)
    new_ops_info = {}
    for node_name in ugraph.topo_order:
      match = self.TARGET_NODENAME_PATTERN.match(node_name)
      if match:
        # ignore all dropout nodes
        continue
      # replace inputs with dropout inputs
      op_info = ugraph.ops_info[node_name]
      in_t_infos = [deepcopy(t_info, {'ugraph': new_graph}) 
                    for t_info in op_info.input_tensors]
      out_t_infos = [deepcopy(t_info, {'ugraph': new_graph}) 
                    for t_info in op_info.output_tensors]
      op_attr = deepcopy(op_info.op_attr)
      for i, t_info in enumerate(in_t_infos):
        op_name = parse_tensor_name(t_info.name)[0]
        match = self.TARGET_NODENAME_PATTERN.match(op_name)
        if match:
          name_scope = match.group(1)
          # assume there should be only on input except keep_prob
          dropout_in_tensor = dropout_input_map[name_scope]
          in_t_infos.pop(i)
          in_t_infos.insert(i, dropout_in_tensor)
      new_op_info = OperationInfo(name=op_info.name,
                                  input_tensors=in_t_infos,
                                  output_tensors=out_t_infos,
                                  op_type=op_info.op_type,
                                  backend=op_info.backend,
                                  op_attr=op_attr,
                                  ugraph=new_graph)
      new_ops_info[node_name] = new_op_info
    new_graph.ops_info = new_ops_info
    new_graph._backend = ugraph._backend
    return new_graph

  def _find_dropout_clusters(self, ugraph):
    clusters = defaultdict(lambda: [])
    for node_name in ugraph.topo_order:
      match = self.TARGET_NODENAME_PATTERN.match(node_name)
      if match:
        name_scope = match.group(1)
        clusters[name_scope].append(node_name)
    return dict(clusters)

  def _find_input(self, ugraph):
    """dropout_name --> input_tensor_info

    input_tensor_info := the tensor info of a tensor which is not generated
                         in the dropout namescope but is consumed by ops in
                         dropout namescope with name not starts with 'keep_prob'
    """
    clusters = self._find_dropout_clusters(ugraph)
    input_map = {}
    for node_name in ugraph.topo_order:
      match = self.TARGET_NODENAME_PATTERN.match(node_name)
      if match:
        name_scope = match.group(1)
        cluster = clusters[name_scope]
        op_info = ugraph.ops_info[node_name]
        for in_tensor_info in op_info.input_tensors:
          in_op_name = parse_tensor_name(in_tensor_info.name)[0]
          if in_op_name not in cluster and not in_op_name.startswith('keep_prob'):
            input_map[name_scope] = in_tensor_info
            # assuming there is only one input for dropout
            break
    return input_map


class BatchNormTransformer(Transformer):
  """Replace Batch Norm namescope with uTensor Op
  """
  METHOD_NAME = 'batch_norm'
  KWARGS_NAMESCOPE = '_batch_norm'
  TARGET_NODENAME_PATTERN = re.compile(r'(BatchNorm[_\w\d]*)/.*')

  def transform(self, ugraph):
    # TODO: implement this!
    pass
