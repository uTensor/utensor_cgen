from utensor_cgen.backend.utensor._graph_lower._op_lower import \
    uTensorRearchGraphLower
from utensor_cgen.backend.utensor.code_generator.rearch import (
    OperatorBase, OperatorFactory)
from utensor_cgen.backend.utensor.snippets.rearch import (DeclareOpSnippet,
                                                          OpConstructSnippet,
                                                          OpEvalSnippet)
from utensor_cgen.legalizer.tflite import TFLiteLegalizer

TFLiteLegalizer.register_op_rename('Tanh', 'TanhOperator')

def handle(op_info):
  op_info.code_gen_attributes['namespaces'] = ('ReferenceOperators',)
uTensorRearchGraphLower.CodgenAttributes.register('TanhOperator', handle)

class TanhEvalSnippet(OpEvalSnippet):
  __inputs__ = ["act_in"]
  __outputs__ = ["act_out"]

@OperatorFactory.register
class TanhOperator(OperatorBase):
  namespaces = ('ReferenceOperators',)
  op_type = 'TanhOperator'

  @classmethod
  def get_constructor_parameters(cls, op_info):
    return tuple()

  @classmethod
  def get_type_signature(cls, op_info):
    return ((op_info.output_tensors[0].dtype,), (op_info.input_tensors[0].dtype,))
  
  def get_declare_snippet(self, op_var_name, with_const_params=True):
    return DeclareOpSnippet(
      self,
      templ_dtypes=[self.out_dtypes[0], self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=self.namespaces,
      with_const_params=with_const_params,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return TanhEvalSnippet(
      op_info,
      templ_dtypes=[self.out_dtypes[0], self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=self.namespaces
    )

  def get_construct_snippet(self, op_var_name):
    return OpConstructSnippet(
      self,
      templ_dtypes=[self.out_dtypes[0], self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=self.namespaces
    )
