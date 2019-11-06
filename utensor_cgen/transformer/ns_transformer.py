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
from utensor_cgen.logger import logger
from utensor_cgen.matcher import uTensorGraphMatcher
from utensor_cgen.utils import (parse_tensor_name, prune_graph,
                                topologic_order_graph)

from .base import Transformer
from .pipeline import TransformerPipeline

__all__ = ["DropoutTransformer", "BatchNormTransformer", "InlineTransformer", "TensorLifeProbe"]


@TransformerPipeline.register_transformer
class TensorLifeProbe(Transformer):
  METHOD_NAME = 'tensorlife'
  KWARGS_NAMESCOPE = '_utensor_utlife'
  DATA_NAME = 'address'

  def __init__(
    self,
    buff_size=100000, #1k bytes
    unit_size=4
  ):
    self.buff_size = buff_size
    self.unit_size = unit_size

  def transform(self, ugraph):
    new_ugraph = deepcopy(ugraph)
    new_ugraph.setup_data_manager({self.DATA_NAME: " "})
    # use_def_table: dict, tensor_name -> {'start': op_idx, 'end': op_idx}
    use_def_table = self._create_resource_table(new_ugraph)
    allocate_table = dict()
    allocate_success = self.allocate_graph(new_ugraph, allocate_table, use_def_table, self.buff_size, self.unit_size)

    if allocate_success:
      for node_name in new_ugraph.topo_order:
        in_t_infos = new_ugraph.ops_info[node_name].input_tensors
        for in_o in in_t_infos:
          if in_o.name in allocate_table:
            new_ugraph.data_manager.address = (in_o.name, allocate_table[in_o.name]['offsetstart'])
        out_t_infos = new_ugraph.ops_info[node_name].output_tensors
        for out_o in out_t_infos:
          if out_o.name in allocate_table:
            new_ugraph.data_manager.address = (out_o.name, allocate_table[out_o.name]['offsetstart'])
      return new_ugraph
    return ugraph

  def _query_offset_fromallocate_table(self, allocate_table, start, end):
    new_start = start
    new_end = end
    for key in allocate_table:
      if allocate_table[key]['offsetstart'] >= start and allocate_table[key]['offsetend'] <= end:
        continue
      elif allocate_table[key]['offsetstart'] <= start and allocate_table[key]['offsetend'] >= start:
        new_start = allocate_table[key]['offsetstart']
        if allocate_table[key]['offsetend'] >= end:
          new_end = max(new_end, allocate_table[key]['offsetend'])
        else:
          new_end = max(end, new_end)
      elif allocate_table[key]['offsetstart'] >= start and allocate_table[key]['offsetend'] >= start:
        if allocate_table[key]['offsetend'] >= end:
          new_end = max(new_end, allocate_table[key]['offsetend'])
        else:
          new_end = max(end, new_end)
    return new_start, new_end

  def _query_time_fromallocate_table(self, allocate_table, start, end):
    time_start = start
    time_end = end
    for key in allocate_table:
      if allocate_table[key]['start'] >= start and allocate_table[key]['end'] <= end:
        continue
      elif allocate_table[key]['start'] <= start and allocate_table[key]['end'] >= start:
        if allocate_table[key]['end'] >= end:
          time_end = max(time_end, allocate_table[key]['end'])
        else:
          time_end = max(end, time_end)
      elif allocate_table[key]['start'] >= start and allocate_table[key]['end'] >= start:
        if allocate_table[key]['end'] >= end:
          time_end = max(time_end, allocate_table[key]['end'])
        else:
          time_end = max(end, time_end)
    return time_start, time_end

  def _query_result(self, allocate_table, offset, length, timestart, timeend):
    for key in allocate_table:
      mem_occupied = (
        (allocate_table[key]['offsetstart'] >= offset and allocate_table[key]['offsetstart'] <= offset + length) or
        (allocate_table[key]['offsetstart'] <= offset and allocate_table[key]['offsetend'] >= offset)
      )
      life_span_occupied = (
        (allocate_table[key]['start'] >= timestart and allocate_table[key]['start'] <= timeend) or
        (allocate_table[key]['start'] <= timestart and allocate_table[key]['end'] >= timestart)
      )
      if mem_occupied and life_span_occupied:
        return True
    return False
  
  def allocate_tensor(self, tensors, tensor_index, allocate_table, use_def_table, buffer_size, unit_size):
    if tensor_index == len(tensors):
      return True
    if tensors[tensor_index].name in allocate_table:
      return self.allocate_tensor(tensors, tensor_index + 1, allocate_table, use_def_table, buffer_size, unit_size)

    tensor = tensors[tensor_index]
    candidates = self._get_candidates(allocate_table, use_def_table, buffer_size, unit_size, tensor)
    if not candidates:
      return False
    success = False
    for candidate in candidates:
      self._update_allocation_table(allocate_table, use_def_table, tensor, candidate, candidate + tensor.size)
      success = self.allocate_tensor(tensors, tensor_index + 1, allocate_table, use_def_table, buffer_size, unit_size)
      if success:
        break
      else:
        self._remove_allocate_table(allocate_table, tensor)
    return success

  def allocate_graph(self, ugraph, allocate_table, use_def_table, buffer_size, unit_size):
    tensors = []

    for node_name in ugraph.topo_order:
      in_t_infos = [
        tensor
        for tensor in ugraph.ops_info[node_name].input_tensors
        if tensor.op.op_type != 'Inline'
      ]
      out_t_infos = [
        tensor
        for tensor in ugraph.ops_info[node_name].output_tensors
        if tensor.op.op_type != 'Inline'
      ]
      tensors.extend(in_t_infos)
      tensors.extend(out_t_infos)
    
    succ = self.allocate_tensor(tensors, 0, allocate_table, use_def_table, buffer_size, unit_size)
    return succ

  def _check(self, allocate_table, use_def_table, tensor, tensor_offset_start, tensor_offset_end):
    valid = False
    timestart = use_def_table[tensor.name]['start']
    timeend = use_def_table[tensor.name]['end']
    offset, length = self._query_offset_fromallocate_table(allocate_table, tensor_offset_start, tensor_offset_end)
    timestart, timeend = self._query_time_fromallocate_table(allocate_table, timestart, timeend)
    occupied = self._query_result(allocate_table, offset, length, timestart, timeend)
    if not occupied:
      valid = True
    return valid

  def _get_candidates(self, allocate_table, use_def_table, buffer_size, unit_size, in_o):
    ret = []
    for i in range(0, buffer_size, unit_size):
      if self._check(allocate_table, use_def_table, in_o, i, i + in_o.size):
        ret.append(i)
    return ret
  
  def _update_allocation_table(
    self,
    allocate_table,
    use_def_table,
    tensor,
    offset_start,
    offset_end
  ):   
    time_start = use_def_table[tensor.name]['start']
    time_end = use_def_table[tensor.name]['end']
    attribute = dict()
    attribute['start'] = time_start
    attribute['end'] = time_end
    attribute['offsetstart'] = offset_start
    attribute['offsetend'] = offset_end
    allocate_table[tensor.name] = attribute
    return allocate_table   
  
  def _remove_allocate_table(self, allocate_table, tensor):
    del allocate_table[tensor.name]

  def _create_resource_table(self, ugraph):
    resource_table = dict()
    len_map = {
      op_name: idx
      for idx, op_name in enumerate(ugraph.topo_order)
    }
    for node_name in ugraph.topo_order:
      for tensor_info in ugraph.ops_info[node_name].input_tensors:
        if tensor_info.name not in resource_table:
          lifetime = dict()
          lifetime['start'] = len_map[node_name]
          lifetime['end'] = len_map[node_name]  
          resource_table[tensor_info.name] = lifetime   
        resource_table[tensor_info.name]['end']= len_map[node_name] 
      
      for outtensor in ugraph.ops_info[node_name].output_tensors:
        if outtensor.name not in resource_table:
          lifetime = dict()
          lifetime['start'] = len_map[node_name]
          lifetime['end'] = len_map[node_name]  
          resource_table[outtensor.name] = lifetime

    return resource_table


@TransformerPipeline.register_transformer
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


@TransformerPipeline.register_transformer
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
          in_op_name = in_tensor_info.op.name
          if in_op_name not in cluster and not in_op_name.startswith('keep_prob'):
            input_map[name_scope] = in_tensor_info
            # assuming there is only one input for dropout
            break
    return input_map


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


@TransformerPipeline.register_transformer
class BatchNormTransformer(Transformer):
  """Replace Batch Norm namescope with uTensor Op
  """
  METHOD_NAME = 'batch_norm'
  KWARGS_NAMESCOPE = '_batch_norm'

  def transform(self, ugraph):
    # TODO: implement this!
    raise RuntimeError('bach norm transformer is not yet implemented')


@TransformerPipeline.register_transformer
class FakeGatherV2Transformer(Transformer):
  """Force converting GatherV2 op to Gather op
  """
  METHOD_NAME = 'fake_gather_v2'
  KWARGS_NAMESCOPE = '_fake_gatherv2'

  def transform(self, ugraph):
    logger.warning(
      "enabling {} will force replacing GatherV2 with Gather".format(self.METHOD_NAME)
    )
    for key, op in ugraph.ops_info.items():
      if op.op_type == "GatherV2":
        op.op_type = "Gather"
        ugraph.ops_info[key] = op
    return ugraph
