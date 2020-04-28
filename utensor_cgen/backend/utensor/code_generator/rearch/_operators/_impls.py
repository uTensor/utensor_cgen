from utensor_cgen.backend.utensor.snippets._types import NP_TYPES_MAP
from utensor_cgen.backend.utensor.snippets.rearch import (AddOpEvalSnippet,
                                                          DeclareOpSnippet)

from ._base import OperatorFactory, _Operator


@OperatorFactory.register
class _AddOperator(_Operator):

  op_type = 'AddOperator'

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    snippet = DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
    )
    return snippet

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    snippet = AddOpEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
    )
    return snippet


@OperatorFactory.register
class _MatmulOperator(_Operator):

  op_type = 'MatrixMultOperator'


@OperatorFactory.register
class _ConvOperator(_Operator):

  op_type = 'ConvOperator'
