from abc import ABCMeta, abstractmethod
from functools import wraps

from utensor_cgen.utils import prune_graph as _prune_graph
from utensor_cgen.utils import topologic_order_graph


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
        return _prune_graph(new_ugraph)
      return new_ugraph

    self.transform = transform
    return self

  def __init__(self, prune_graph=True, **kwargs):
    # just for defining the __init__ signature
    pass

  @abstractmethod
  def transform(self, ugraph):
    raise NotImplementedError('You should overwrite transform method for all transformer')
