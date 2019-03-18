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
class TensorLifeProbe(Transformer):
  METHOD_NAME = 'tensorlife'
  KWARGS_NAMESCOPE = '_utensor_utlife'


  def transform(self, ugraph):
    use_def_table = self._create_resource_table(ugraph)
    unit_size = 8
    allocate_success = False
    buffer_size = 1000 #1k bytes
    while not allocate_success:
      allocate_table = dict()
      for node_name in ugraph.topo_order:
        '''in_t_infos = ugraph.input_tensors[node_name] may be input tensor doest not need this
        for in_o in in_t_infos:
          success = False
          for i in range(0, buffer_size, unit_size):
            if self._check(ugraph, allocate_table, use_def_table, in_o, i, i + in_o.shape):
              tensor_size = self._getSize(in_o.shape)
              self._create_allocate_table(allocate_table, use_def_table, in_o, i, i + tensor_size)
              success = True 
            if success == False:
              break
        '''
        out_t_infos = ugraph.output_tensors[node_name]
        for out_o in out_t_infos:
          success = False
          for i in range(0, buffer_size, unit_size):
            if self._check(ugraph, allocate_table, use_def_table, out_o, i, i + out_o.shape):
              tensor_size = self._getSize(out_o.shape)
              self._create_allocate_table(allocate_table, use_def_table, out_o, i, i + tensor_size)
              success = True
            if success == False:
              break
            
      allocate_success = True


  def _query_offset_fromallocate_table(self, allocate_table, start, end):
    new_start = start
    new_end = end
    for key, value in allocate_table:
      if value['offsetstart'] >= start and value['offsetend'] <= end:
        continue
      elif value['offsetstart'] <= start and value['offsetend'] >= start:
        new_start = key
        if value['offsetend'] >= end:
          new_end = max(new_end, value['offsetend'])
        else:
          new_end = max(end, new_end)
      elif value['offsetstart'] >= start and value['offsetend'] >= start:
        if value['offsetend'] >= end:
          new_end = max(new_end, value['offsetend'])
        else:
          new_end = max(end, new_end)
    return new_start, new_end

  def _query_time_fromallocate_table(self, allocate_table, op, offset, length, timestart,
  timeend):
    ret = list()
    for key, value in allocate_table:
      if (key >= offset and key < offset + length) or (key < offset and key + length < offset + length):
        if value['timestart'] <= timestart and value['timeend'] >= timestart:
          ret.append(op)
        elif value['timestart'] >= timestart and value['timestart'] < timeend:
          ret.append(op)
    return ret

  def _check(self, ugraph, allocate_table, use_def_table, op, op_offset_start, op_offset_end):
    valid = False
    timestart = use_def_table[op.name].start
    timeend = use_def_table[op.name].end
    offset, length = self._query_offset_fromallocate_table(allocate_table, op_offset_start, op_offset_end)
    a = self._query_time_fromallocate_table(allocate_table, op, offset, length, timestart, timeend)
    if len(a) == 0:
      return True
    return valid

  def _create_allocate_table(self, resource_table, use_def_table, tensor,
  offset_start, offset_end):   
    time_start = use_def_table[tensor.name].start
    time_end = use_def_table[tensor.name].end
    attribute = dict()
    attribute['start'] = time_start
    attribute['end'] = time_end
    attribute['offsetstart'] = offset_start
    attribute['offsetend'] = offset_end
    resource_table[tensor.name] = attribute
    return resource_table   

  def _create_resource_table(self, ugraph):
    resource_table = dict()
    len_map = self._create_length(ugraph)
    for node_name in ugraph.topo_order:
      #op_info_t = ugraph.ops_info[node_name]
      #for op_info in ugraph.ops_info():
      for tensor_info in ugraph.ops_info.input_tensors:
        if not resource_table[tensor_info]:
          lifetime = dict()
          lifetime['start'] = len_map[node_name]
          lifetime['end'] = len_map[node_name]  
          resource_table[tensor_info.name] = lifetime   
        resource_table[tensor_info.name]['end']= len_map[node_name] 
      
      for outtensor in ugraph.ops_info.output_tensors:
        if not resource_table[outtensor.name]:
          lifetime = dict()
          lifetime['start'] = len_map[node_name]
          lifetime['end'] = len_map[node_name]  
          resource_table[outtensor.name] = lifetime
  
    return resource_table
  
  def _create_length(self, ugraph):
    my_map = dict()
    index = 0
    for name in ugraph.topo_order:
      my_map[name] = index
      index = index + 1
    return my_map
  def _getSize(self, tensor_shape):
    ret = 1
    for i in tensor_shape:
      ret = ret * i
    return ret

class InlineTransformer(Transformer):
  METHOD_NAME = 'inline'
  KWARGS_NAMESCOPE = '_utensor_inline'
  TARGET_NODENAME_PATTERN = re.compile(r'(const[_\w\d]*)/.*')


  def transform(self, ugraph):
    for node_name in ugraph.topo_order:
      op_type = ugraph.ops_info[node_name].op_type
      op_inputs = ugraph.tensor
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
