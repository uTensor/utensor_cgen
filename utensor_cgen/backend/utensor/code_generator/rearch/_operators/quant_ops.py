from utensor_cgen.backend.utensor.snippets.rearch import DeclareOpSnippet
from utensor_cgen.utils import MUST_OVERWRITE

from .base import OperatorFactory, _OperatorBase


class _QuantOperator(_OperatorBase):
  op_type = MUST_OVERWRITE

  def __init__(self, op_info):
    cls = type(self)
    if cls.op_type is MUST_OVERWRITE:
      raise ValueError('must overwrite op_type: {}'.format(cls))


@OperatorFactory.register
class _QuantAddOperator(_QuantOperator):
  op_type = 'QuantAddOperator'

  def get_declare_snippet(self, op_var_name):
    # TODO: quant add declaration snippet
    pass

  def get_eval_snippet(self, op_info, op_name, tensor_var_map):
    # TODO: quant add eval snippet
    pass
