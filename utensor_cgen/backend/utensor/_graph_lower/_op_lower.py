from copy import deepcopy

from utensor_cgen.backend.base import BackendPart
from utensor_cgen.logger import logger
from utensor_cgen.utils import Configuration, class_property


class uTensorGraphLowerBase(BackendPart):
  TARGET = 'utensor'

  def handle_default(self, ugraph):
    logger.warning('fall back to default graph lowering (do nothing)')
    return ugraph
  
  def get_handler(self, ugraph):
    handler = getattr(self, 'handle_{}'.format(ugraph.lib_name), self.handle_default)
    return handler
  
  def apply(self, ugraph):
    handler = self.get_handler(ugraph)
    return handler(ugraph)

class uTensorLegacyGraphLower(uTensorGraphLowerBase):
  PART = 'legacy_graph_lower'

  def handle_tensorflow(self, ugraph):
    for op_info in ugraph.get_ops_by_type('AddOperator'):
      op_info.op_type = 'Add'
    return ugraph

  @class_property
  def default_config(cls):
    return {}


class uTensorRearchGraphLower(uTensorGraphLowerBase):
  PART = 'rearch_graph_lower'

  class OptypeRenameManager(object):
    NAME_MAP = {
      'Add': 'AddOperator',
      'Conv2D': 'ConvOperator',
      'MatMul': 'MatrixMultOperator'
    }

    @classmethod
    def get_new_optype(cls, op_type):
      return cls.NAME_MAP.get(op_type, op_type)
  
  class CheckQuantization(object):

    @classmethod
    def apply(cls, ugraph):
      if cls._check_quantized(ugraph):
        for op_info in ugraph.get_ops_by_type('DepthwiseSeparableConvOperator'):
          op_info.op_type = 'QuantizedDepthwiseSeparableConvOperator'
        for op_info in ugraph.get_ops_by_type('FullyConnectedOperator'):
          op_info.op_type = 'QuantizedFullyConnectedOperator'
      for op_info in ugraph.get_ops_by_type('DequantizeOperator'):
        op_info.code_gen_attributes['namespaces'] = ('TFLM',)
      for op_info in ugraph.get_ops_by_type('QuantizeOperator'):
        op_info.code_gen_attributes['namespaces'] = ('TFLM',)
    
    @classmethod
    def _check_quantized(cls, ugraph):
      for op_info in ugraph.ops_info.values():
        for tensor_info in op_info.output_tensors:
          # FIXME: better way to check quantization
          if 'quantization_zeros' in tensor_info.attributes:
            return True
  
  @classmethod
  def add_name_map(cls, generic_name, target_specific_name):
    cls.OptypeRenameManager.NAME_MAP[generic_name] = target_specific_name

  def handle_tensorflow(self, ugraph):
    for op_info in ugraph.ops_info.values():
      op_info.op_type = self.OptypeRenameManager.get_new_optype(op_info.op_type)
 
  def handle_tflite(self, ugraph):
    self.CheckQuantization.apply(ugraph)
  
  @class_property
  def default_config(cls):
    return {}
