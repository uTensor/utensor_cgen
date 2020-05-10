from copy import deepcopy
from functools import reduce

from utensor_cgen.ir.base import OperationInfo
from utensor_cgen.utils import topologic_order_graph

from .api import Legalizer
from .base import LegalizerBase


@Legalizer.register
class TFLiteLegalizer(LegalizerBase):
  TARGET = "tflite"

  def legalize_ops(self, ugraph):
    _GraphRewrite.apply(ugraph)
    _OpTypeRename.apply(ugraph)
    return ugraph
  
  def legalize_dtype(self, ugraph):
    return ugraph


class _OpTypeRename(object):
  _OPTYPE_RENAME_MAP = {
    "FullyConnected": "QuantizedFullyConnectedOperator",
    "Quantize": "QuantizeOperator",
    "DepthwiseConv2d": "QuantizedDepthwiseSeparableConvOperator",
    "MaxPool2d": "MaxPoolOperator",
    "Dequantize": "DequantizeOperator",
    "Reshape": "ReshapeOperator",
  }
  
  @classmethod
  def apply(cls, ugraph):
    for op_info in ugraph.ops_info.values():
      op_info.op_type = cls._OPTYPE_RENAME_MAP.get(
        op_info.op_type,
        op_info.op_type,
      )


class _GraphRewrite(object):

  @classmethod
  def apply(cls, ugraph):
    # 1. transpose the filter to make a right mulitiplication: fc = x @ filter + bias
    # 2. if the input is not flatten, inject a reshape op
    reshape_cnt = 0
    for op_info in ugraph.get_ops_by_type('FullyConnected'):
      filter_tensor = op_info.input_tensors[1]
      filter_op = filter_tensor.op
      np_arr = filter_op.op_attr['value'].value.np_array
      filter_op.op_attr['value'].value.np_array = np_arr.T
      filter_tensor.shape = list(np_arr.T.shape)
      filter_op.output_tensors[0].shape = list(np_arr.T.shape)

      tensor_x = op_info.input_tensors[0]
      if len(tensor_x.shape) > 2:
        new_shape = [tensor_x.shape[0], reduce(lambda a, b: a*b, tensor_x.shape[1:], 1)]
        reshape_op_name = tensor_x.name.replace(":", "_") + '_Reshape' + str(reshape_cnt)
        out_tensor = deepcopy(tensor_x, {'ugraph': ugraph})
        out_tensor.name = reshape_op_name + ":0"
        out_tensor.op_name = reshape_op_name
        out_tensor.shape = new_shape
        OperationInfo(
          name=reshape_op_name,
          op_type="Reshape",
          lib_name='tflite',
          ugraph=ugraph,
          input_tensors=[tensor_x],
          output_tensors=[out_tensor],
          op_attr={
            'new_shape': new_shape
          }
        )
        reshape_cnt += 1
        op_info.input_tensors[0] = out_tensor
    topologic_order_graph(ugraph)
