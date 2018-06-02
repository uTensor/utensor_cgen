from .base import Transformer

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy

__all__ = ['OptimizerFactory']


class RefCntOptimizer(Transformer):

  def transform(self, ugraph):
    """Optimization with reference count
    """
    self.prune_graph = False
    return self._transform(ugraph)
  
  def _transform(self, ugraph):
    new_graph = deepcopy(ugraph)
    refcnt_table = _tensor_ref_count(new_ugraph.ops_info)
    for op_name in new_graph.topo_order[::-1]:
      op_info = new_graph.ops_info[op_name]
      if op_name in ugraph.output_nodes or op_info.op_type in ["Const", "Placeholder"]:
        op_info.op_attr['_optimize__to_eval'] = False
      else:
        op_info.op_attr['_optimize__to_eval'] = True
      ref_counts = dict((t_info.name, refcnt_table[t_info.name]) for t_info in op_info.output_tensors)
      op_info.op_attr['_optimize__ref_counts'] = ref_counts
    return new_graph

  @staticmethod
  def _tensor_ref_count(ops_info):
    tensor_ref_count = defaultdict(lambda: 0)
    for op_info in ops_info.values():
      for tensor_info in op_info.input_tensors:
        tname = tensor_info.name
        tensor_ref_count[tname] += 1
    return tensor_ref_count
  
class NonOptimizer(Transformer):

  def transform(self, ugraph):
    self.prune_graph = False
    return ugraph


class OptimizerFactory(Transformer):

  _OPTIMIZE_METHODS = {
    'None': NonOptimizer
    'refcnt': RefCntOptimizer
  }

  def __init__(self, method, *args, **kwargs):
    self._delegate = self._OPTIMIZE_METHODS[method](*args, **kwargs)
  
  def transform(self, ugraph):
    return self._delegate.transform(ugraph)

  @classmethod
  def get_optimizer(cls, method='None', **kwargs):
    default = cls._OPTIMIZE_METHODS['None']
    optimzer = cls._OPTIMIZE_METHODS.get(method, default)(**kwargs)
    return optimzer