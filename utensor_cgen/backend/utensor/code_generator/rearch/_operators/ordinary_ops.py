from utensor_cgen.backend.utensor.snippets._types import NP_TYPES_MAP
from utensor_cgen.backend.utensor.snippets.rearch import (AddOpEvalSnippet,
                                                          DeclareOpSnippet,
                                                          RomTensorSnippet)
from utensor_cgen.utils import MUST_OVERWRITE

from .base import OperatorFactory, _OperatorBase


class _Operator(_OperatorBase):
  op_type = MUST_OVERWRITE

  def __init__(self, op_info):
    cls = type(self)
    if cls.op_type is MUST_OVERWRITE:
      raise ValueError('must overwrite op_type: {}'.format(cls))

@OperatorFactory.register
class _AddOperator(_Operator):

  op_type = 'AddOperator'

  def get_declare_snippet(self, op_var_name):
    snippet = DeclareOpSnippet(
      op_type=self.op_type,
      dtypes=[NP_TYPES_MAP[self.in_dtypes[0]].tensor_type_str],
      op_var_name=op_var_name,
    )
    return snippet

  def get_eval_snippet(self, op_info, op_name, tensor_var_map):
    snippet = AddOpEvalSnippet(
      op_info=op_info,
      op_name=op_name,
      tensor_var_map=tensor_var_map,
      dtypes=[op_info.input_tensors[0].dtype]
    )
    return snippet


@OperatorFactory.register
class _MatmulOperator(_Operator):

  op_type = 'MatrixMultOperator'


@OperatorFactory.register
class _ConvOperator(_Operator):

  op_type = 'ConvOperator'


@OperatorFactory.register
class _InlineOperator(_Operator):

  op_type = 'Inline'

  def __init__(self, op_info):
    self._tensor = op_info.output_tensors[0]

  def get_declare_snippet(self, tensor_var_name, buffer_var_name, tensor):
    snippet = RomTensorSnippet(
      tensor_var_name=tensor_var_name,
      buffer_var_name=buffer_var_name,
      tensor=tensor
    )
    return snippet
