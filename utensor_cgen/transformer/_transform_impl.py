from abc import ABCMeta, abstractmethod
from collections import defaultdict
import re

from ..parser import OperationInfo, TensorInfo
from ..parser.utils import parse_tensor_name


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
          for in_tensor in ops_info[name].input_tensor:
            tname = in_tensor[0]
            in_op_name, _ = parse_tensor_name(tname)
            if in_op_name not in cluster_ops and ns not in input_ops_map:
              input_ops_map[ns] = ops_info[in_op_name].output_tensor[0]
              break
    return input_ops_map

  # def _get_output_map(self, ops_info, dropout_clusters):
  #   """
  #   return dict
  #   dropout_name -> output_tensor_name
  #   """
  #   output_ops_map = {}
  #   for op_name in ops_info:
  #     for input_tensor_info in ops_info[op_name].input_tensor:
  #       in_tname = input_tensor_info.name
  #       in_op_name, _ = parse_tensor_name(in_tname)
  #       match = self._NAMESCOPE_PATTERN.match(in_op_name)
  #       if match:
  #         ns = match.group(1)
  #         if op_name not in dropout_clusters[ns] and ns not in output_ops_map:
  #           output_ops_map[ns] = input_tensor_info.name
  #           break
  #   return output_ops_map

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
    pass


class Transformer(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def transform(self, ops_info, topo_orders, output_nodes):
    raise NotImplementedError('You should overwrite transform method for all transformer')


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
