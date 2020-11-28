from .api import Legalizer
from .base import LegalizerBase
from .utils import _hotfix_reshape


@Legalizer.register
class GraphDefLegalizer(LegalizerBase):
  TARGET = 'tensorflow'

  class _OpTypeRenamePostProcessing(object):
    _RENAME_MAP = {
      'BatchMatMulV2': 'MatMul',
      # FIXME: need to update matcher before adding this line
      'Add': 'AddOperator',
      'ArgMax': 'ArgMaxOperator',
      'Dequantize': 'DequantizeOperator',
      'Max': 'MaxOperator',
      'Min': 'MinOperator',
      'MaxPool':'MaxPoolOperator',
      'MatMul': 'FullyConnectedOperator',
      'Relu': 'ReLUOperator',
      'Reshape': 'ReshapeOperator',
      'Conv2D': 'Conv2dOperator',
      'Const': "Constant",
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
    for op_info in ugraph.get_ops_by_type('FullyConnectedOperator'):
      op_info.op_attr['FusedActivationFunction'] = '0 (NONE)'
    for op_info in ugraph.get_ops_by_type('Conv2dOperator'):
      op_attr = op_info.op_attr
      op_attr['Padding'] = {
        b'VALID': 1,
        b'SAME': 2,
      }[op_attr['padding'].value]
      _, op_attr['StrideW'], op_attr['StrideH'], _ = op_attr['strides'].value.ints_value
      del op_attr['padding'], op_attr['strides']
    ugraph = _hotfix_reshape(ugraph)
    return ugraph

  def legalize_dtype(self, ugraph):
    '''Legalize data types of tensors in given graph
    '''
    if not ugraph.lib_name == self.TARGET:
      raise ValueError(
        'expecting tensorflow graph, get {}'.format(ugraph.lib_name)
      )
    return ugraph
