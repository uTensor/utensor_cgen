# -*- coding:utf8 -*-
r"""Namescope Transformer

Transformers that get rid of namescope/nodes which are not needed 
for inference
"""
import re
from copy import deepcopy

import numpy as np

import tensorflow as tf
from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.matcher import uTensorGraphMatcher
from utensor_cgen.utils import prune_graph, topologic_order_graph

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

  @property
  def pattern_ugraph(self):
    graph = tf.Graph()
    with graph.as_default():
        dummy_x = tf.constant(np.random.rand(10), dtype=tf.float32, name='dummy_x')
        dummy_rate = tf.constant(0.5, dtype=tf.float32, name='dummy_rate')
        dropout = tf.nn.dropout(dummy_x, rate=dummy_rate, name='dropout')
    patrn_ugraph = GraphDefParser.parse(graph.as_graph_def(), output_nodes=[dropout.op.name])
    patrn_ugraph['dropout/truediv'].replace_with_null_input_tensor(0) # replace dummy_x
    patrn_ugraph['dropout/sub'].replace_with_null_input_tensor(1) # replce dummy_rate
    patrn_ugraph = prune_graph(patrn_ugraph)
    topologic_order_graph(patrn_ugraph)
    return patrn_ugraph
  
  def transform(self, ugraph):
    new_ugraph = deepcopy(ugraph)
    if new_ugraph.backend == 'tensorflow':
      new_ugraph = self._transform_tf(new_ugraph)
    else:
      raise ValueError(
        'only support dropout transformer for tensorflow: get {}'.format(new_ugraph.backend)
      )
    return new_ugraph
  
  def _transform_tf(self, ugraph):
    matcher = uTensorGraphMatcher(pattern_ugraph=self.pattern_ugraph)
    matches = matcher.match(ugraph, n=1)
    while matches:
      match = matches[0]
      ugraph = self._handle_match_tf(match)
      matches = matcher.match(ugraph)
    return ugraph
  
  def _handle_match_tf(self, match):
    subj_ugraph = match.subject_ugraph
    subj_in_tensor = (
      match.patrn2subj_op_map['dropout/truediv']
      .input_tensors[0]
      .op
      .output_tensors[0]
    )
    subj_out_op = match.patrn2subj_op_map['dropout/mul']
    subj_out_tensor = subj_out_op.output_tensors[0]
    for op in subj_out_op.output_nodes:
      for idx, tensor in enumerate(op.input_tensors):
        if tensor.name == subj_out_tensor.name:
          op.input_tensors[idx] = subj_in_tensor
    match.subject_ugraph = prune_graph(subj_ugraph)
    topologic_order_graph(match.subject_ugraph)
    return match.subject_ugraph


class BatchNormTransformer(Transformer):
  """Replace Batch Norm namescope with uTensor Op
  """
  METHOD_NAME = 'batch_norm'
  KWARGS_NAMESCOPE = '_batch_norm'
  TARGET_NODENAME_PATTERN = re.compile(r'(BatchNorm[_\w\d]*)/.*')

  def transform(self, ugraph):
    # TODO: implement this!
    pass


class FakeGatherV2Transformer(Transformer):
  """Force converting GatherV2 op to Gather op
  """
  METHOD_NAME = 'FakeGatherV2'
  KWARGS_NAMESCOPE = '_fake_gatherv2'
  TARGET_NODENAME_PATTERN = re.compile(r'(GatherV2[_\w\d]*)/.*')

  def transform(self, ugraph):
    print("warning: force replacing GatherV2 with Gather")
    for key, op in ugraph.ops_info.items():
      if op.op_type == "GatherV2":
        op.op_type = "Gather"
        ugraph.ops_info[key] = op
    return ugraph
