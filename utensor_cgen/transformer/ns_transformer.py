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

__all__ = ["DropoutTransformer", "BatchNormTransformer", "InlineTransformer", "TensorLifeProbe"]

class TensorLifeProbe(Transformer):
  METHOD_NAME = 'tensorlife'
  KWARGS_NAMESCOPE = '_utensor_utlife'


  def transform(self, ugraph):
    use_def_table = self._create_resource_table(ugraph)
    unit_size = 4
    allocate_success = False
    buffer_size = 50 #1k bytes
    while not allocate_success:
      allocate_table = dict()
      for node_name in ugraph.topo_order:

        success = False
        in_t_infos = ugraph.ops_info[node_name].input_tensors  # may be input tensor doest not need this
        for in_o in in_t_infos:
          for i in range(0, buffer_size, unit_size):
            tensor_size = self._getSize(in_o)
            if self._check(allocate_table, use_def_table, in_o, i, i + tensor_size):
              self._create_allocate_table(allocate_table, use_def_table, in_o, i, i + tensor_size)
              success = True
              break
            else:
              success = False 
        
        out_t_infos = ugraph.ops_info[node_name].output_tensors
        for out_o in out_t_infos:
          success = False
          for i in range(0, buffer_size, unit_size):
            tensor_size = self._getSize(out_o)
            if self._check(allocate_table, use_def_table, out_o, i, i + tensor_size):
              self._create_allocate_table(allocate_table, use_def_table, out_o, i, i + tensor_size)
              success = True
              break
            else:
              success = False
        if success == False:
          break
      if success == True:
        allocate_success = True
    for node_name in ugraph.topo_order:
      in_t_infos = ugraph.ops_info[node_name].input_tensors
      for in_o in in_t_infos:
        if in_o.name in allocate_table:
          in_o.alloc_address[0] = allocate_table[in_o.name]['offsetstart']
          in_o.alloc_address[1] = allocate_table[in_o.name]['offsetend']
      out_t_infos = ugraph.ops_info[node_name].output_tensors
      for out_o in out_t_infos:
        if out_o.name in allocate_table:
          out_o.alloc_address[0] = allocate_table[out_o.name]['offsetstart']
          out_o.alloc_address[1] = allocate_table[out_o.name]['offsetend']
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

  def _query_result(self, allocate_table, op, offset, length, timestart,
  timeend):
    ret = list()
    for key in allocate_table:
      if (allocate_table[key]['offsetstart'] >= offset and allocate_table[key]['offsetstart'] <= offset + length) or (allocate_table[key]['offsetstart'] <= offset and allocate_table[key]['offsetend'] >= offset):
        if (allocate_table[key]['start'] >= timestart and allocate_table[key]['start'] <= timeend) or (allocate_table[key]['start']
        <= timestart and allocate_table[key]['end'] >= timeend):
          ret.append(op)

    return ret

  def _check(self, allocate_table, use_def_table, tensor, tensor_offset_start, tensor_offset_end):
    valid = False
    timestart = use_def_table[tensor.name]['start']
    timeend = use_def_table[tensor.name]['end']
    offset, length = self._query_offset_fromallocate_table(allocate_table, tensor_offset_start, tensor_offset_end)
    timestart, timeend = self._query_time_fromallocate_table(allocate_table, timestart, timeend)
    a = self._query_result(allocate_table, tensor, offset, length, timestart, timeend)
    if len(a) == 0:
      return True
    return valid

  def _create_allocate_table(self, allocate_table, use_def_table, tensor,
  offset_start, offset_end):   
    time_start = use_def_table[tensor.name]['start']
    time_end = use_def_table[tensor.name]['end']
    attribute = dict()
    attribute['start'] = time_start
    attribute['end'] = time_end
    attribute['offsetstart'] = offset_start
    attribute['offsetend'] = offset_end
    allocate_table[tensor.name] = attribute
    return allocate_table   

  def _create_resource_table(self, ugraph):
    resource_table = dict()
    len_map = self._create_topo_list(ugraph)
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
  
  def _create_topo_list(self, ugraph):
    my_map = {node_idx: name for (node_idx, name) in enumerate(ugraph.topo_order)}
    return my_map
  
  #should consider different type
  def _getSize(self, tensor):
    ret = 1   
    for i in tensor.shape:
      ret = ret * i
    return ret


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
