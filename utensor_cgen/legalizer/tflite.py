from .base import LegalizerBase


class TFLiteLegalizer(LegalizerBase):
  TARGET = "tflite"

  class _OpTypeRename(object):
    _OPTYPE_RENAME_MAP = {
      "FullyConnected": "QuantizedFullyConnectedOperator",
      "Quantize": "QuantizeOperator",
      "DepthwiseConv2d": "QuantizedDepthwiseSeparableConvOperator",
      "MaxPool2d": "MaxPoolOperator",
      "Dequantize": "DequantizeOperator",
    }
    
    @classmethod
    def apply(cls, ugraph):
      for op_info in ugraph.ops_info.values():
        op_info.op_type = cls._OPTYPE_RENAME_MAP.get(
          op_info.op_type,
          op_info.op_type,
        )

  def legalize_ops(self, ugraph):
    self._OpTypeRename.apply(ugraph)
    for op_info in ugraph.get_ops_by_type('QuantizedFullyConnectedOperator'):
      filter_tensor = op_info.input_tensors[1]
      filter_op = filter_tensor.op
      assert filter_tensor is filter_op.output_tensors[0]
      np_arr = filter_op.op_attr['value'].value.np_array.reshape(filter_tensor.shape)
      filter_op.op_attr['value'].value.np_array = np_arr.T
      filter_tensor.shape = list(np_arr.T.shape)
    return ugraph
  
  def legalize_dtype(self, ugraph):
    return ugraph
