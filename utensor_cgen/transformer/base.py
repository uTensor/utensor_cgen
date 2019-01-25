from abc import ABCMeta, abstractmethod
from copy import deepcopy
from functools import wraps

from utensor_cgen.utils import parse_tensor_name, topologic_order_graph


class Transformer(object):
  """
  Any subclass of Transformer should follow the 
  signature of __new__ and __init__ defined here
  if the overwrite these two methods in order to 
  work properly
  """
  __metaclass__ = ABCMeta
  KWARGS_NAMESCOPE = None
  METHOD_NAME = None

  def __new__(cls,
              prune_graph=True,
              **kwargs):
    if cls.KWARGS_NAMESCOPE is None:
      raise ValueError('kwargs namescope not found for %s' % cls)
    self = object.__new__(cls)
    self.prune_graph = prune_graph
    ori_transform = self.transform

    @wraps(ori_transform)
    def transform(ugraph):
      new_ugraph = ori_transform(ugraph)
      topologic_order_graph(new_ugraph)
      if self.prune_graph:
        return self._prune_graph(new_ugraph)
      return new_ugraph

    self.transform = transform
    return self

  def __init__(self, prune_graph=True, **kwargs):
    # just for define the __init__ signature
    pass

  @abstractmethod
  def transform(self, ugraph):
    raise NotImplementedError('You should overwrite transform method for all transformer')

  @classmethod
  def _prune_graph(cls, ugraph):
    """Remove nodes that is no longer needed
    """
    new_ugraph = deepcopy(ugraph)
    # BFS to find all ops you need
    ops_in_need = set(ugraph.output_nodes)
    queue = [name for name in ugraph.output_nodes]
    visited = set([])
    while queue:
      op_name = queue.pop(0)
      #FIXME: names are framework and graph dependent
      #       temporary fix is included aftert the commented code
      #op_info = new_ugraph.ops_info[op_name]
      # in_ops = [parse_tensor_name(t_info.name)[0] 
      #           for t_info in op_info.input_tensors]

      #TODO: move the code below to a standalone function. Consider using a more extensive data structure:
      #      Or, use this: in_ops = [node.name for node in ugraph.ops_info[op_name].input_nodes]
      tensors_in = set([t.name for t in ugraph.ops_info[op_name].input_tensors])
      in_ops = set()
      for it_node in ugraph.topo_order:
        if(it_node == op_name):
          continue
        it_tensors_out = [t.name for t in ugraph.ops_info[it_node].output_tensors]
        if not tensors_in.isdisjoint(it_tensors_out):
          in_ops.add(it_node)

      queue.extend([name for name in in_ops if name not in visited])
      visited.update(in_ops)
      ops_in_need.update(in_ops)
      #END

    ops_to_remove = set([])
    for op_name in new_ugraph.ops_info.keys():
      if op_name not in ops_in_need:
        # remove ops not needed from ops_info
        ops_to_remove.add(op_name)
    for op_name in ops_to_remove:
      new_ugraph.ops_info.pop(op_name)
    return new_ugraph
