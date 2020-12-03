from copy import deepcopy
from itertools import chain

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


class uTensorRearchGraphLower(uTensorGraphLowerBase):
  PART = 'rearch_graph_lower'
  
  class CodgenAttributes(object):

    _HANDLERS = {}

    @classmethod
    def register(cls, op_type, func=None):
      def regist_handle(handler):
        cls._HANDLERS[op_type] = handler
        return handler
      if func:
        return regist_handle(func)
      return regist_handle

    @classmethod
    def apply(cls, ugraph):
      # TODO: better abstraction, sth like lowering strategy
      for op_info in ugraph.get_ops_by_type("AddOperator"):
        op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type("MulOperator"):
        op_info.code_gen_attributes["namespaces"] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type("SinOperator"):
        op_info.code_gen_attributes["namespaces"] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type("TransposeOperator"):
        op_info.code_gen_attributes["namespaces"] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type("ReshapeOperator"):
        op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type("MatrixMultOperator"):
        op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type('ArgMinOperator'):
        op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type('ArgMaxOperator'):
        op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type('QuantizeOperator'):
        op_info.code_gen_attributes['namespaces'] = ('TflmSymQuantOps',)
      for op_info in ugraph.get_ops_by_type('DequantizeOperator'):
        op_info.code_gen_attributes['namespaces'] = ('TflmSymQuantOps',)
      for op_info in ugraph.get_ops_by_type('ReLUOperator'):
        op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type('ReLU6Operator'):
        op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type('MinOperator'):
        op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type('MaxOperator'):
        op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type('AvgPoolOperator'):
        op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type('MaxPoolOperator'):
        op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type('MinPoolOperator'):
        op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type('Conv2dOperator'):
        op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type('DepthwiseSeparableConvOperator'):
        if cls._check_quantized(op_info):
          op_info.code_gen_attributes['namespaces'] = ('TflmSymQuantOps',)
        else:
          op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_info in ugraph.get_ops_by_type('FullyConnectedOperator'):
        if cls._check_quantized(op_info):
          op_info.code_gen_attributes['namespaces'] = ('TflmSymQuantOps',)
        else:
          op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
      for op_type, handler in cls._HANDLERS.items():
        for op_info in ugraph.get_ops_by_type(op_type):
          handler(op_info)

    @classmethod
    def _check_quantized(cls, op_info):
      for tensor_info in chain(
        op_info.output_tensors,
        op_info.input_tensors
      ):
        # FIXME: better way to check quantization
        if 'quantization_zeros' in tensor_info.attributes:
          return True
      return False
  
  @classmethod
  def add_name_map(cls, generic_name, target_specific_name):
    cls.OptypeRenameManager.NAME_MAP[generic_name] = target_specific_name

  def handle_tensorflow(self, ugraph):
    self.CodgenAttributes.apply(ugraph)
 
  def handle_tflite(self, ugraph):
    self.CodgenAttributes.apply(ugraph)
  
  @class_property
  def default_config(cls):
    return {}
