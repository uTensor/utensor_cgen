from utensor_cgen.logger import logger

from .api import Legalizer
from .base import LegalizerBase


@Legalizer.register
class GraphDefLegalizer(LegalizerBase):
  TARGET = 'tensorflow'

  class _OpTypeRenamePostProcessing(object):
    _RENAME_MAP = {
      'BatchMatMulV2': 'FullyConnectedOperator',
      'Add': 'AddOperator',
      'MatMul': 'FullyConnectedOperator',
      'Relu': 'ReLUOperator',
      'Conv2D': 'Conv2dOperator',
      'ArgMax': 'ArgMaxOperator',
      'Reshape': 'ReshapeOperator',
      'MaxPool': 'MaxPoolOperator',
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

    # old TF fully connected layer does not fuse activation at all
    for op_info in ugraph.get_ops_by_type("FullyConnectedOperator"):
      op_info.op_attr["FusedActivationFunction"] = "0 (NONE)"
    for op_info in ugraph.get_ops_by_type("Conv2dOperator"):
      padding = op_info.op_attr["padding"].value.decode('utf8')
      op_info.op_attr["Padding"] = {
        "VALID": 1,
        "SAME": 2
      }[padding]
      _, stride_w, stride_h, _ = op_info.op_attr["strides"].value.ints_value
      op_info.op_attr['StrideW'] = stride_w
      op_info.op_attr['StrideH'] = stride_h
    for op_info in ugraph.get_ops_by_type("ReshapeOperator"):
      shape_op = op_info.input_tensors.pop(-1).op
      new_shape = shape_op.op_attr['value'].value.np_array.tolist()
      if new_shape[0] == -1:
        temp = new_shape[:]
        new_shape[0] = 1
        logger.warning(f'nondeterministic batch size detected, make the batch size to 1: {op_info.name}, {temp} -> {new_shape}')
      op_info.op_attr['new_shape'] = new_shape
      op_info.n_inputs -= 1
      del ugraph.ops_info[shape_op.name]
    return ugraph

  def legalize_dtype(self, ugraph):
    '''Legalize data types of tensors in given graph
    '''
    if not ugraph.lib_name == self.TARGET:
      raise ValueError(
        'expecting tensorflow graph, get {}'.format(ugraph.lib_name)
      )
    return ugraph
