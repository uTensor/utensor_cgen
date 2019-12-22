from utensor_cgen.backend.base import BackendPart
from utensor_cgen.utils import class_property


class uTensorGraphLower(BackendPart):

  TARGET = 'utensor'
  PART = 'graph_lower'

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
