from copy import deepcopy

from utensor_cgen.backend.base import BackendPart
from utensor_cgen.utils import class_property


class uTensorLegacyGraphLower(BackendPart):

  TARGET = 'utensor'
  PART = 'legacy_graph_lower'

  def apply(self, ugraph):
    handler = getattr(self, 'handle_{}'.format(ugraph.lib_name))
    if handler is None:
      raise RuntimeError(
        'can not lower ugraph from {} to utensor'.format(ugraph.lib_name)
      )
    return handler(ugraph)

  def handle_tensorflow(self, ugraph):
    return ugraph

  @class_property
  def default_config(cls):
    return {}


class uTensorRearchGraphLower(BackendPart):
  TARGET = 'utensor'
  PART = 'rearch_graph_lower'

  class OptypRenameManager(object):
    NAME_MAP = {
      'Add': 'AddOperator',
      'Conv2D': 'ConvOperator',
      'MatMul': 'MatrixMultOperator'
    }

    @classmethod
    def get_new_optype(cls, op_type):
      return cls.NAME_MAP.get(op_type, op_type)

  def apply(self, ugraph):
    handler = getattr(self, 'handle_{}'.format(ugraph.lib_name))
    if handler is None:
      raise RuntimeError(
        'can not lower ugraph from {} to utensor'.format(ugraph.lib_name)
      )
    return handler(ugraph)

  def handle_tensorflow(self, ugraph):
    new_ugraph = deepcopy(ugraph)
    for op_info in new_ugraph.ops_info.values():
      op_info.op_type = self.OptypRenameManager.get_new_optype(op_info.op_type)
    return new_ugraph
  
  @class_property
  def default_config(cls):
    return {}
  

