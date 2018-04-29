# -*- coding:utf8 -*-
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from functools import wraps
import re

from ..parser import OperationInfo, TensorInfo
from ..parser.utils import parse_tensor_name

__all__ = ["NamescopeTransformer", "Transformer"]


class _DropoutTransformer(object):
  """Remove Dropout Op
  """
  _NAMESCOPE_PATTERN = re.compile(r'(dropout[_\w\d]*)/.*')

  def transform(self, ops_info, topo_orders, output_nodes):
    """
    ops_info : dict
    topo_orders : List[str]
      operation name in topological sorted order
    output_nodes : List[str]
      output node names
    """
    (new_ops_info,
     new_topo_orders,
     new_output_nodes) = self._transform(ops_info, topo_orders, output_nodes)
    return new_ops_info, new_topo_orders, new_output_nodes

  def _find_dropout_ns(self, ops_info, topo_orders):
    dropout_ns = defaultdict(lambda: [])
    for name in topo_orders:
      match = self._NAMESCOPE_PATTERN.match(name)
      if match:
        ns = match.group(1)
        dropout_ns[ns].append(name)
    return dropout_ns

  def _get_input_map(self, ops_info, dropout_clusters):
    """
    return dict
    dropout_name -> input_tensor_info
    """
    input_ops_map = {}
    for op_name in ops_info:
      match = self._NAMESCOPE_PATTERN.match(op_name)
      if match:
        ns = match.group(1)
        cluster_ops = dropout_clusters[ns]
        for name in cluster_ops:
          for in_tensor_info in ops_info[name].input_tensor:
            tname = in_tensor_info.name
            in_op_name, _ = parse_tensor_name(tname)
            if in_op_name not in cluster_ops and ns not in input_ops_map:
              in_op_info = ops_info[in_op_name]
              # NOTE ignore the input if the name starts with "keep_prob" and
              # it's a placeholder
              if in_op_info.node_name.startswith('keep_prob') and \
                 in_op_info.op_type == 'Placeholder':
                 continue
              input_ops_map[ns] = ops_info[in_op_name].output_tensor[0]
    return input_ops_map

  def _transform(self, ops_info, topo_orders, output_nodes):
    # TODO return new ops_info, topo_orders and output_nodes
    dropout_ns = self._find_dropout_ns(ops_info, topo_orders)
    input_map = self._get_input_map(ops_info, dropout_ns)
    new_ops_info = {}
    new_topo_orders = []
    for op_name in topo_orders:
      match = self._NAMESCOPE_PATTERN.match(op_name)
      # ignore matched name (droupout)
      if not match:
        op_info = ops_info[op_name]
        new_input_tensor = []
        # find and replace drouput output
        for in_tensor_info in op_info.input_tensor:
          in_op_name, _ = parse_tensor_name(in_tensor_info.name)
          match = self._NAMESCOPE_PATTERN.match(in_op_name)
          if match:
            ns = match.group(1)
            dp_input_tensor_info = input_map[ns]
            new_tensor_info = dp_input_tensor_info
          else:
            new_tensor_info = TensorInfo(name=in_tensor_info.name,
                                         dtype=in_tensor_info.dtype,
                                         shape=in_tensor_info.shape)
          new_input_tensor.append(new_tensor_info)
        # update op_info and topo orders
        new_op_info = OperationInfo(node_name=op_info.node_name,
                                    input_tensor=new_input_tensor,
                                    output_tensor=op_info.output_tensor,
                                    op_type=op_info.op_type,
                                    output_content=op_info.output_content,
                                    op_attr=op_info.op_attr)
        new_ops_info[op_name] = new_op_info
        new_topo_orders.append(op_name)
    return new_ops_info, new_topo_orders, output_nodes


class _BatchNormTransformer(object):
  """Replace Batch Norm namescope with uTensor Op
  """

  _NAMESCOPE_PATTERN = re.compile(r'(BatchNorm[_\w\d]*)/.*')

  def transform(self, ops_info, topo_orders, output_nodes):
    # TODO implement this!
    pass


class Transformer(object):
  __metaclass__ = ABCMeta

  # NOTE maybe move the pruning to metaclasss is a cleaner solution?
  # though the function closure works well here.
  def __new__(cls, *args, **kwargs):
    self = object.__new__(cls)
    ori_transform = self.transform

    @wraps(ori_transform)
    def transform(ops_info, topo_orders, output_nodes):
      # after each transformation, we prune the graph.
      # what we do with pruning is like removing unneeded nodes
      # or clean up garbages
      (new_ops_info,
       new_topo_orders,
       new_output_nodes) = ori_transform(ops_info,
                                         topo_orders,
                                         output_nodes)
      (new_ops_info_clean,
       new_topo_orders_clean,
       new_output_nodes_clean) = self.prune_graph(new_ops_info,
                                                  new_topo_orders,
                                                  new_output_nodes)
      return (new_ops_info_clean,
              new_topo_orders_clean,
              new_output_nodes_clean)
    self.transform = transform
    return self

  @abstractmethod
  def transform(self, ops_info, topo_orders, output_nodes):
    raise NotImplementedError('You should overwrite transform method for all transformer')

  def prune_graph(self, ops_info, topo_orders, output_nodes):
    ops_info, topo_orders, output_nodes = self._clean_up_placeholders(ops_info,
                                                                      topo_orders,
                                                                      output_nodes)
    return ops_info, topo_orders, output_nodes

  def _clean_up_placeholders(self, ops_info, topo_orders, output_nodes):
    # remove placeholders which output to nowhere
    new_ops_info = {}
    new_topo_orders = []
    for this_op_name in topo_orders:
      this_op_info = ops_info[this_op_name]
      if this_op_info.op_type == "Placeholder":
        if not self._is_needed(this_op_name, ops_info, topo_orders):
          continue
      new_ops_info[this_op_name] = this_op_info
      new_topo_orders.append(this_op_name)
    return new_ops_info, new_topo_orders, output_nodes

  def _is_needed(self, this_op_name, ops_info, topo_orders):
    this_op_info = ops_info[this_op_name]
    out_op_names = [parse_tensor_name(tensor_info.name)[0]
                    for tensor_info in this_op_info.output_tensor]
    for out_op_name in out_op_names:
      for op_name in filter(lambda name: name != this_op_name, topo_orders):
        in_tensor_infos = ops_info[op_name].input_tensor
        in_tnames = [parse_tensor_name(tensor_info.name)[0] for tensor_info in in_tensor_infos]
        if out_op_name in in_tnames:
          return True
    return False


class NamescopeTransformer(Transformer):

  _DELEGATION_MAP = {
    "dropout": _DropoutTransformer,
    "BatchNorm": _BatchNormTransformer
  }

  def __init__(self, target_name_scope):
    if target_name_scope not in self._DELEGATION_MAP:
      raise ValueError('Unsupport namescope: {}'.format(target_name_scope))
    self._target_ns = target_name_scope
    self._delegate = self._DELEGATION_MAP[target_name_scope]()

  def transform(self, ops_info, topo_orders, output_nodes):
    (new_ops_info,
     new_topo_orders,
     new_output_nodes) = self._delegate.transform(ops_info, topo_orders, output_nodes)
    return new_ops_info, new_topo_orders, new_output_nodes

  def __repr__(self):
    return "NamescopeTransformer('{}')".format(self._target_ns)
