#-*- coding:utf8 -*-
from collections import defaultdict


class Optimizer(object):

  @classmethod
  def optimize(cls, ops_info, bfs_order, output_nodes, level='simple'):
    """
    [(op_name, op_info, ref_counts, to_eval), ...]
    """
    if level == 'simple':
      return cls._simple_optimize(ops_info, bfs_order, output_nodes)
    else:
      raise ValueError("unknown optimization level: {}".format(level))

  @classmethod
  def _simple_optimize(cls, ops_info, bfs_order, output_nodes):
    """Optimization using only reference count
    """
    optimized_order = []

    refcnt_table = cls._tensor_ref_count(ops_info)
    for op_name in bfs_order[::-1]:
      op_info = ops_info[op_name]
      if op_name in output_nodes or op_info.op_type in ["Const", "Placeholder"]:
        to_eval = False
      else:
        to_eval = True
      ref_counts = [refcnt_table[out_tname] for out_tname, _, _ in op_info.output_tensor]
      optimized_order.append((op_name, op_info, ref_counts, to_eval))
    return optimized_order

  @classmethod
  def _tensor_ref_count(cls, ops_info):
    tensor_ref_count = defaultdict(lambda: 0)
    for op_info in ops_info.values():
      for tname, _, _ in op_info.input_tensor:
        tensor_ref_count[tname] += 1
    return tensor_ref_count
