from abc import ABCMeta, abstractmethod
from functools import wraps

from utensor_cgen.logger import logger
from utensor_cgen.utils import prune_graph as _prune_graph
from utensor_cgen.utils import topologic_order_graph

GENERIC_SENTINEL = object()

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
  APPLICABLE_LIBS = set()

  def __init__(self, prune_graph=True, **kwargs):
    cls = type(self)
    if not cls.APPLICABLE_LIBS:
      logger.warning(
        "empty APPLICABLE_LIBS detected for {}, ".format(cls) + \
        "such transformer will not be applied to any graph by default"
      )
    if cls.KWARGS_NAMESCOPE is None:
      raise ValueError('kwargs namescope not found for %s' % cls)
    self.prune_graph = prune_graph
    ori_transform = self.transform

    @wraps(ori_transform)
    def transform(ugraph):
      if self.APPLICABLE_LIBS is not GENERIC_SENTINEL and ugraph.lib_name not in self.APPLICABLE_LIBS:
        logger.info(
          "%s is not applicable to ugraph with lib name %s, skipping",
          self,
          ugraph.lib_name,
        )
        return ugraph
      new_ugraph = ori_transform(ugraph)
      topologic_order_graph(new_ugraph)
      if self.prune_graph:
        return _prune_graph(new_ugraph)
      return new_ugraph

    self.transform = transform

  @abstractmethod
  def transform(self, ugraph):
    raise NotImplementedError('You should overwrite transform method for all transformer')
  
  @classmethod
  def mark_applicable(cls, *lib_names):
    cls.APPLICABLE_LIBS.update(lib_names)
    return cls
