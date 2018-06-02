from abc import ABCMeta, abstractmethod
from functools import wraps


class Transformer(object):
  __metaclass__ = ABCMeta

  def __new__(cls,
              *args,
              prune_graph=True,
              **kwargs):
    self = object.__new__(cls)
    self.prune_graph = prune_graph
    ori_transform = self.transform

    @wraps(ori_transform)
    def transform(ugraph):
      new_graph = ori_transform(ugraph)
      if self.prune_graph:
        return self._prune_graph(new_graph)
      return new_graph

    self.transform = transform
    return self

  @abstractmethod
  def transform(self, ugraph):
    raise NotImplementedError('You should overwrite transform method for all transformer')

  @classmethod
  def _prune_graph(cls, ugraph):
    """Remove nodes that is no longer needed
    """
    new_ugraph = deepcopy(ugraph)
    # BFS to find all ops you need
    all_in_ops = set(ugraph.output_nodes)
    queue = [name for name in ugraph.output_nodes]
    visited = set([])
    while queue:
      op_name = queue.pop(0)
      op_info = new_ugraph.ops_info[op_name]
      in_ops = [parse_tensor_name(tname)[0] 
                for tname in op_info.input_tensors]
      queue.extend([name for name in in_ops if name not in visited])
      visited.update(in_ops)
      all_in_ops.update(in_ops)
    new_ops_info = {}
    new_topo_order = []
    for i, op_name in enumerate(new_ugraph.new_topo_order):
      if op_name not in all_in_ops:
        # remove ops not needed
        new_ugraph.ops_info.pop(op_name)
        new_ugraph.topo_order.pop(i)
