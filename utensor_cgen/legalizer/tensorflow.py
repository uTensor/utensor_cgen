from .api import Legalizer
from .base import LegalizerBase


@Legalizer.register
class GraphDefLegalizer(LegalizerBase):
  TARGET = 'tensorflow'

  class _OpTypeRenamePostProcessing(object):
    _RENAME_MAP = {
      'BatchMatMulV2': 'MatMul',
      # 'Add': 'AddOperator', # FIXME: need to update matcher before adding this line
    }

    @classmethod
    def apply(cls, ugraph):
      for op_type, new_op_type in cls._RENAME_MAP.items():
        for op_info in ugraph.get_ops_by_type(op_type):
          op_info.op_type = new_op_type

  def legalize_ops(self, ugraph):
    '''Legalize ops to generic ops in given graph
    '''
    if not ugraph.lib_name == self.TARGET:
      raise ValueError(
        'expecting tensorflow graph, get {}'.format(ugraph.lib_name)
      )
    self._OpTypeRenamePostProcessing.apply(ugraph)
    return ugraph

  def legalize_dtype(self, ugraph):
    '''Legalize data types of tensors in given graph
    '''
    if not ugraph.lib_name == self.TARGET:
      raise ValueError(
        'expecting tensorflow graph, get {}'.format(ugraph.lib_name)
      )
    return ugraph
