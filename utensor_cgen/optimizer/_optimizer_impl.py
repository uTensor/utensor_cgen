#-*- coding:utf8 -*-
from abc import ABCMeta, abstractmethod
from collections import defaultdict

__all__ = ['RefCntOptimizer']


class Optimizer(object):
  __metaclass__ = ABCMeta

  @classmethod
  @abstractmethod
  def optimize(cls, ops_info, topo_order, output_nodes):
    raise NotImplementedError('You should overwrite optimize method for all optimzer')


class RefCntOptimizer(Optimizer):

  @classmethod
  def optimize(cls, ops_info, topo_order, output_nodes, method='None'):
    """
    [(op_name, op_info, ref_counts, to_eval), ...]
    """
    opt_function = _OPTIMIZE_METHODS.get(method, None)
    if opt_function is None:
      raise ValueError("unknown optimization method: {}".format(method))
    return opt_function(ops_info, topo_order, output_nodes)


def _no_optimize(ops_info, topo_order, output_nodes):
  optimized_order = []
  for op_name in topo_order[::-1]:
    op_info = ops_info[op_name]
    ref_cnts = [0 for _ in op_info.output_tensor]
    optimized_order.append((op_name, op_info, ref_cnts, False))
  return optimized_order


def _refcnt_optimize(ops_info, topo_order, output_nodes):
  """Optimization using only reference count
  """
  optimized_order = []

  refcnt_table = _tensor_ref_count(ops_info)
  for op_name in topo_order[::-1]:
    op_info = ops_info[op_name]
    if op_name in output_nodes or op_info.op_type in ["Const", "Placeholder"]:
      to_eval = False
    else:
      to_eval = True
    ref_counts = [refcnt_table[out_tname] for out_tname in
                  [tensor_info.name for tensor_info in op_info.output_tensor]]
    optimized_order.append((op_name, op_info, ref_counts, to_eval))
  return optimized_order


def _tensor_ref_count(ops_info):
  tensor_ref_count = defaultdict(lambda: 0)
  for op_info in ops_info.values():
    for tensor_info in op_info.input_tensor:
      tname = tensor_info.name
      tensor_ref_count[tname] += 1
  return tensor_ref_count


_OPTIMIZE_METHODS = {
  'None': _no_optimize,
  'refcnt': _refcnt_optimize
}
