# -*- coding:utf8 -*-
r"""Namescope Transformer

Transformers that get rid of namescope/nodes which are not needed 
for inference
"""
import re
from collections import defaultdict
from copy import deepcopy

import numpy as np
import tensorflow.compat.v1 as tf

# FIXME: remove uTensorOpEqualityDelegate import after we have generic ops
from utensor_cgen.backend.utensor.code_generator.legacy._operators import \
    uTensorOpEqualityDelegate
from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.ir import OperationInfo, uTensorGraph
from utensor_cgen.logger import logger
from utensor_cgen.matcher import uTensorGraphMatcher
from utensor_cgen.utils import (parse_tensor_name, prune_graph,
                                topologic_order_graph)

from .base import GENERIC_SENTINEL, Transformer
from .pipeline import TransformerPipeline

__all__ = [
  "DropoutTransformer",
  "BatchNormTransformer",
  "InlineTransformer",
]


@TransformerPipeline.register_transformer
class BiasAddTransformer(Transformer):
  METHOD_NAME = 'biasAdd'
  KWARGS_NAMESCOPE = '_utensor_biasAdd'
  APPLICABLE_LIBS = GENERIC_SENTINEL

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


@TransformerPipeline.register_transformer
class InlineTransformer(Transformer):
  METHOD_NAME = 'inline'
  KWARGS_NAMESCOPE = '_utensor_inline'
  APPLICABLE_LIBS = GENERIC_SENTINEL

  def __init__(self):
    super(InlineTransformer, self).__init__(prune_graph=False)

  def transform(self, ugraph):
    for node_name in ugraph.topo_order:
      op_type = ugraph.ops_info[node_name].op_type
      if op_type == 'Const':
        op_info = ugraph.ops_info[node_name]
        op_info.op_type = 'Inline'
    
    return ugraph


@TransformerPipeline.register_transformer
class DropoutTransformer(Transformer):
  """Dropout removal transformer

  Pros
  ====
  - Insensitive to the dropout layer pattern so it works across different
    versions of tensorflow

  Cons
  ====
  - naming constrains on the dropout layers, layer name must matched to the
    given `name_pattern` (default to r'(dropout[_\w\d]*)/.*') and the keep_prob
    op must be with name starts with 'keep_prop'
  """
  METHOD_NAME = 'dropout'
  KWARGS_NAMESCOPE = '_utensor_dropout'
  APPLICABLE_LIBS = GENERIC_SENTINEL

  def __init__(self, name_pattern=r'(dropout[_\w\d]*)/.*'):
    super(DropoutTransformer, self).__init__(prune_graph=True)
    self._op_name_pattern = re.compile(name_pattern)

  def transform(self, ugraph):
    new_ugraph = deepcopy(ugraph)
    (
      dropout_input_map,
      dropout_output_map,
      clusters
     ) = self._find_dropout_clusters(ugraph)
    for name_scope, cluster in clusters.items():
      input_tensor = list(dropout_input_map[name_scope])[0].output_tensors[0]
      output_tensor = list(dropout_output_map[name_scope])[0].output_tensors[0]
      for node in output_tensor.op.output_nodes:
        new_input = new_ugraph.ops_info[input_tensor.op.name].output_tensors[0]
        new_node = new_ugraph.ops_info[node.name]
        for i, tensor in enumerate(new_node.input_tensors):
          if tensor.name == output_tensor.name:
            new_node.input_tensors[i] = new_input
      for op_info in cluster:
        if op_info.name in new_ugraph.ops_info:
          del new_ugraph.ops_info[op_info.name]
    return new_ugraph

  def _find_dropout_clusters(self, ugraph):
    clusters = defaultdict(set)
    for op_info in ugraph.ops_info.values():
      match = self._op_name_pattern.match(op_info.name)
      if match:
        name_scope = match.group(1)
        clusters[name_scope].add(op_info)
    clusters = dict(clusters)
    input_map = defaultdict(set)
    output_map = defaultdict(set)
    for name_scope, cluster in clusters.items():
      for op_info in cluster:
        for tensor in op_info.input_tensors:
          if not re.match(r"^(rate|keep_prob).*", tensor.op.name) and tensor.op not in cluster:
            input_map[name_scope].add(tensor.op)
        for out_node in op_info.output_nodes:
          if out_node not in cluster:
            output_map[name_scope].add(op_info)
    return dict(input_map), dict(output_map), clusters


@TransformerPipeline.register_transformer
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
  APPLICABLE_LIBS = set(["tensorflow"])

  @property
  def pattern_ugraph(self):
    graph = tf.Graph()
    with graph.as_default():
        dummy_x = tf.constant(np.random.rand(10, 10), dtype=tf.float32, name='dummy_x')
        dummy_rate = tf.placeholder(dtype=tf.float32, name='dummy_rate')
        dropout = tf.nn.dropout(dummy_x, rate=dummy_rate, name='dropout')
    patrn_ugraph = GraphDefParser(config={}).parse(graph.as_graph_def(), output_nodes=[dropout.op.name])
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
    if new_ugraph.lib_name == 'tensorflow':
      new_ugraph = self._transform_tf(new_ugraph)
    else:
      raise ValueError(
        'dropout transformer only support tensorflow graph, get {}'.format(new_ugraph.lib_name)
      )
    return new_ugraph
  
  def _transform_tf(self, ugraph):
    # FIXME: should use a generic op_equality_delegate
    matcher = uTensorGraphMatcher(
      pattern_ugraph=self.pattern_ugraph,
      op_equality_delegate=uTensorOpEqualityDelegate
    )
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


@TransformerPipeline.register_transformer
class BatchNormTransformer(Transformer):
  """Replace Batch Norm namescope with uTensor Op
  """
  METHOD_NAME = 'batch_norm'
  KWARGS_NAMESCOPE = '_batch_norm'
  APPLICABLE_LIBS = GENERIC_SENTINEL

  def transform(self, ugraph):
    # TODO: implement this!
    raise RuntimeError('bach norm transformer is not yet implemented')


@TransformerPipeline.register_transformer
class FakeGatherV2Transformer(Transformer):
  """Force converting GatherV2 op to Gather op
  """
  METHOD_NAME = 'fake_gather_v2'
  KWARGS_NAMESCOPE = '_fake_gatherv2'
  APPLICABLE_LIBS = GENERIC_SENTINEL

  def transform(self, ugraph):
    logger.warning(
      "enabling {} will force replacing GatherV2 with Gather".format(self.METHOD_NAME)
    )
    for key, op in ugraph.ops_info.items():
      if op.op_type == "GatherV2":
        op.op_type = "Gather"
        ugraph.ops_info[key] = op
    return ugraph
