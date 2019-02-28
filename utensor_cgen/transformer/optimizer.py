from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy

from .base import Transformer

__all__ = ['RefCntOptimizer']


class RefCntOptimizer(Transformer):
  
  METHOD_NAME = 'refcnt'
  KWARGS_NAMESCOPE = '_utensor_refcnt'

  def __init__(self, **kwargs):
    self.prune_graph = False

  def transform(self, ugraph):
    """Optimization with reference count
    """
    return self._transform(ugraph)
  
  def _transform(self, ugraph):
    new_ugraph = deepcopy(ugraph)
    refcnt_table = self._tensor_ref_count(new_ugraph.ops_info)
    for op_name in new_ugraph.topo_order[::-1]:
      op_info = new_ugraph.ops_info[op_name]
      if op_name in ugraph.output_nodes or op_info.op_type in ["Const", "Placeholder"]:
        op_info.op_attr['%s__to_eval' % self.KWARGS_NAMESCOPE] = False
      else:
        op_info.op_attr['%s__to_eval' % self.KWARGS_NAMESCOPE] = True
      ref_counts = [refcnt_table[t_info.name] for t_info in op_info.output_tensors]
      op_info.op_attr['%s__ref_counts' % self.KWARGS_NAMESCOPE] = ref_counts
    return new_ugraph

  @staticmethod
  def _tensor_ref_count(ops_info):
    tensor_ref_count = defaultdict(lambda: 0)
    for op_info in ops_info.values():
      for tensor_info in op_info.input_tensors:
        tname = tensor_info.name
        tensor_ref_count[tname] += 1
    return tensor_ref_count

class IdOpRemoveOptimizer(Transformer):

  METHOD_NAME = 'remove_id_op'
  KWARGS_NAMESCOPE = '_utensor_remove_id_op'

  def __init__(self, **kwargs):
    self.prune_graph = True

  def transform(self, ugraph):
    if ugraph.backend == 'tensorflow':
      return self._transform_tf(ugraph)
    else:
      raise RuntimeError('unsupported backend: {}'.format(ugraph.backend))

  def _transform_tf(self, ugraph):
    ops_to_remove = [
      op
      for op_name, op in ugraph.ops_info.items()
      if op.op_type == 'Identity' 
    ]
    for op in ops_to_remove:
      in_tensor = op.input_tensors[0]
      out_tensor = op.output_tensors[0]
      for out_node in op.output_nodes:
        new_input_tensors = []
        for tensor in out_node.input_tensors:
          if tensor.name == out_tensor.name:
            new_input_tensors.append(in_tensor)
          else:
            new_input_tensors.append(tensor)
        out_node.input_tensors = new_input_tensors
    return ugraph
