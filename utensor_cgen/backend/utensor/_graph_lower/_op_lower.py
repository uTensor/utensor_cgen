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
    return ugraph

  @class_property
  def default_config(cls):
    return {}


class uTensorRearchGraphLower(uTensorGraphLowerBase):
  PART = 'rearch_graph_lower'

  def __init__(self, config):
    final_config = Configuration(self.default_config, config)
    self.tflite_use_quant_dws_conv = final_config['tflite_use_quant_dws_conv']

  class OptypeRenameManager(object):
    NAME_MAP = {
      'Add': 'AddOperator',
      'Conv2D': 'ConvOperator',
      'MatMul': 'MatrixMultOperator'
    }

    @classmethod
    def get_new_optype(cls, op_type):
      return cls.NAME_MAP.get(op_type, op_type)
  
  class AddCodegenAttributes(object):

    @classmethod
    def add_attributes(cls, ugraph):
      for op_info in ugraph.get_ops_by_type('DepthwiseSeparableConvOperator'):
        op_info.code_gen_attributes['namespaces'] = ('TFLM',)
  
  @classmethod
  def add_name_map(cls, generic_name, target_specific_name):
    cls.OptypeRenameManager.NAME_MAP[generic_name] = target_specific_name

  def handle_tensorflow(self, ugraph):
    for op_info in ugraph.ops_info.values():
      op_info.op_type = self.OptypeRenameManager.get_new_optype(op_info.op_type)
 
  def handle_tflite(self, ugraph):
    if self.tflite_use_quant_dws_conv:
      self.AddCodegenAttributes.add_attributes(ugraph)
  
  @class_property
  def default_config(cls):
    return {
      'tflite_use_quant_dws_conv': True,
    }
