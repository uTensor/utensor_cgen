from utensor_cgen.backend.utensor._graph_lower._op_lower import \
    uTensorRearchGraphLower
from utensor_cgen.backend.utensor.code_generator.rearch import (
    OperatorBase, OperatorFactory)
from utensor_cgen.backend.utensor.snippets.rearch import (DeclareOpSnippet,
                                                          OpConstructSnippet,
                                                          OpEvalSnippet)
from utensor_cgen.legalizer.tflite import TFLiteLegalizer

TFLiteLegalizer.register_op_rename('Tanh', 'TanhOperator')

# You may add handle for op_info of specific type with register decorator
@uTensorRearchGraphLower.CodgenAttributes.register('TanhOperator')
def handle(op_info):
  op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
# or call it as method
# uTensorRearchGraphLower.CodgenAttributes.register('TanhOperator', handle)

class TanhEvalSnippet(OpEvalSnippet):
  # TODO: setup __inputs__/__outputs__
  __inputs__ = []
  __outputs__ = []

@OperatorFactory.register
class TanhOperator(OperatorBase):
  namespaces = ('ReferenceOperators',)
  op_type = 'TanhOperator'

  @classmethod
  def get_constructor_parameters(cls, op_info):
    pass

  @classmethod
  def get_type_signature(cls, op_info):
    pass
  
  def get_declare_snippet(self, op_var_name, with_const_params=True):
    pass

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    pass

  def get_construct_snippet(self, op_var_name):
    pass
