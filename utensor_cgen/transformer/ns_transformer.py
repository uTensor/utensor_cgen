# -*- coding:utf8 -*-
r"""Namescope Transformer

Transformers that get rid of namescope/nodes which are not needed 
for inference
"""
import re
from collections import defaultdict
from copy import deepcopy

import numpy as np

import tensorflow as tf
from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.ir import OperationInfo, uTensorGraph
from utensor_cgen.matcher import uTensorGraphMatcher
from utensor_cgen.utils import (parse_tensor_name, prune_graph,
                                topologic_order_graph)

from .base import Transformer

__all__ = ["DropoutTransformer", "BatchNormTransformer", "InlineTransformer"]


class BiasAddTransformer(Transformer):
  METHOD_NAME = 'biasAdd'
  KWARGS_NAMESCOPE = '_utensor_biasAdd'

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

  def transform(self, ugraph):
    for node_name in ugraph.topo_order:
      op_type = ugraph.ops_info[node_name].op_type
      if op_type == 'Const':
        op_info = ugraph.ops_info[node_name]
        op_info.op_type = 'Inline'
    
    return ugraph


class DropoutTransformer(Transformer):
  """Dropout removal transformer

  Pros
  ====
  - Insensitive to the dropout layer pattern so it works across different
    versions of tensorflow

  Cons
  ====
  - naming constrains on the dropout layers, layer name must starts with 'dropout'
    and the keep_prob op must be with name starts with 'keep_prop'
  """
  METHOD_NAME = 'dropout'
  KWARGS_NAMESCOPE = '_utensor_dropout'
  TARGET_NODENAME_PATTERN = re.compile(r'(dropout[_\w\d]*)/.*')

  def __init__(self, name_pattern=r'(dropout[_\w\d]*)/.*'):
    self._op_name_pattern = re.compile(name_pattern)

  def transform(self, ugraph):
    new_graph = uTensorGraph(output_nodes=ugraph.output_nodes)
    dropout_input_map = self._find_input(ugraph)
    new_ops_info = {}
    for node_name in ugraph.ops_info:
      match = self._op_name_pattern.match(node_name)
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
        match = self._op_name_pattern.match(op_name)
        if match:
          name_scope = match.group(1)
          # assume there should be only on input except keep_prob
          dropout_in_tensor = dropout_input_map[name_scope]
          in_t_infos.pop(i)
          in_t_infos.insert(i, dropout_in_tensor)
      new_op_info = OperationInfo(name=op_info.name,
                                  input_tensors=in_t_infos,
                                  n_inputs=len(in_t_infos),
                                  output_tensors=out_t_infos,
                                  n_outputs=len(out_t_infos),
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
      match = self._op_name_pattern.match(node_name)
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
      match = self._op_name_pattern.match(node_name)
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


class DropoutTransformerV2(Transformer):
  """Dropout removal transformer version 2

  Implemented with subgraph matcher

  Pros
  ====
  - no naming requirements on the dropout layer and keep prob op

  Cons
  ====
  - sensitive to the dropout layer pattern. The pattern of dropout
    layer may differ across different version of tensorflow so this
    transformer may fail to match the dropout layer if the given graph
    is not using the same version
  """
  METHOD_NAME = 'dropout_v2'
  KWARGS_NAMESCOPE = '_utensor_dropout_v2'

  @property
  def pattern_ugraph(self):
    graph = tf.Graph()
    with graph.as_default():
        dummy_x = tf.constant(np.random.rand(10, 10), dtype=tf.float32, name='dummy_x')
        dummy_rate = tf.placeholder(dtype=tf.float32, name='dummy_rate')
        dropout = tf.nn.dropout(dummy_x, rate=dummy_rate, name='dropout')
    patrn_ugraph = GraphDefParser.parse(graph.as_graph_def(), output_nodes=[dropout.op.name])
    # replace dummy_x
    patrn_ugraph['dropout/truediv'].replace_with_null_input_tensor(0)
    # # replace dummy_rate
    patrn_ugraph['dropout/sub'].replace_with_null_input_tensor(1)
    # # replace Shape Op
    patrn_ugraph['dropout/random_uniform/RandomUniform'].replace_with_null_input_tensor(0)
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
    for idx, op_name in enumerate(subj_ugraph.output_nodes):
      if op_name == subj_out_op.name:
        subj_ugraph.output_nodes[idx] = subj_in_tensor.op_name
    match.subject_ugraph = prune_graph(subj_ugraph)
    topologic_order_graph(match.subject_ugraph)
    return match.subject_ugraph


class BatchNormTransformer(Transformer):
  """Replace Batch Norm namescope with uTensor Op
  """
  METHOD_NAME = 'batch_norm'
  KWARGS_NAMESCOPE = '_batch_norm'

  def transform(self, ugraph):
    # TODO: implement this!
    pass


class FakeGatherV2Transformer(Transformer):
  """Force converting GatherV2 op to Gather op
  """
  METHOD_NAME = 'FakeGatherV2'
  KWARGS_NAMESCOPE = '_fake_gatherv2'

  def transform(self, ugraph):
    print("warning: force replacing GatherV2 with Gather")
    for key, op in ugraph.ops_info.items():
      if op.op_type == "GatherV2":
        op.op_type = "Gather"
        ugraph.ops_info[key] = op
    return ugraph
