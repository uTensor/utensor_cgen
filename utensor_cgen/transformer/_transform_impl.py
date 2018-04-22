from abc import ABCMeta, abstractmethod
from collections import defaultdict
import re

from ..parser.utils import parse_tensor_name


class _DropoutFuseTransformer(object):

  _PATTERN = re.compile(r'(dropout[_\w\d]*)/.*')

  def transform(self, ops_info, topo_orders, output_nodes):
    """
    ops_info : dict
    topo_orders : List[str]
      operation name in topological sorted order
    output_nodes : List[str]
      output node names
    """
    return self._transform(ops_info, topo_orders, output_nodes)

  def _find_dropout_ns(self, ops_info, topo_orders):
    dropout_ns = defaultdict(lambda: [])
    for name in topo_orders:
      match = self._PATTERN.match(name)
      if match:
        ns = match.group(1)
        dropout_ns[ns].append(name)
    return dropout_ns

  def _find_input_ops(self, ops_info, dropout_clusters):
    """find op name which are
    1. it's input of ops in `cluster_ops`
    2. it's not found in `cluster_ops`
    """
    input_ops_map = defaultdict(lambda: set([]))
    for op_name in ops_info:
      match = self._PATTERN.match(op_name)
      if match:
        ns = match.group(1)
        cluster_ops = dropout_clusters[ns]
        for name in cluster_ops:
          for in_tensor in ops_info[name].input_tensor:
            tname = in_tensor[0]
            in_op_name, _ = parse_tensor_name(tname)
            if in_op_name not in cluster_ops:
              input_ops_map[ns].add(in_op_name)
    return dict(input_ops_map)

  def _find_output_ops(self, ops_info, dropout_clusters):
    output_ops_map = defaultdict(lambda: set([]))
    for op_name in ops_info:
      for input_tensor in ops_info[op_name].input_tensor:
        input_tname = input_tensor[0]
        in_op_name, _ = parse_tensor_name(input_tname)
        match = self._PATTERN.match(in_op_name)
        if match:
          ns = match.group(1)
          if op_name not in dropout_clusters[ns]:
            output_ops_map[ns].add(in_op_name)
    return dict(output_ops_map)

  def _transform(self, ops_info, topo_orders, output_nodes):
    # TODO return new ops_info, topo_orders and output_nodes
    return ops_info, topo_orders, output_nodes


class _BatchNormFuseTransformer(object):

  _PATTERN = re.compile(r'(BatchNorm[_\w\d]*)/.*')

  def transform(self, ops_info, topo_orders, output_nodes):
    pass


class Transformer(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def transform(self, ops_info, topo_orders, output_nodes):
    raise NotImplementedError('You should overwrite transform method for all transformer')


class FuseOpTransformer(Transformer):

  _DELEGATION_MAP = {
    "dropout": _DropoutFuseTransformer,
    "BatchNorm": _BatchNormFuseTransformer
  }

  def __init__(self, target_name_scope):
    if target_name_scope not in self._DELEGATION_MAP:
      raise ValueError('Unsupport fuse type: {}'.format(target_name_scope))
    self._target_ns = target_name_scope
    self._delegate = self._DELEGATION_MAP[target_name_scope]()

  def transform(self, ops_info, topo_orders, output_nodes):
    new_outputs = self._delegate.transform(ops_info, topo_orders, output_nodes)
    return new_outputs
