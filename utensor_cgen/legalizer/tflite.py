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
    return ugraph
  
  def legalize_dtype(self, ugraph):
    return ugraph
