from copy import deepcopy

from six import with_metaclass

from utensor_cgen.backend.utensor.snippets._types import NP_TYPES_MAP
from utensor_cgen.backend.utensor.snippets.rearch import (AddOpEvalSnippet,
                                                          DeclareOpSnippet,
                                                          RomTensorSnippet)

__all__ = ['OperatorFactory', 'OpNotSupportedError']


class OpNotSupportedError(Exception): pass


class OperatorFactory(object):

  _operators = {}

  @classmethod
  def get_opertor(cls, op_info):
    op_type = op_info.op_type
    op_cls = cls._operators.get(op_type)
    if op_cls is None:
      raise OpNotSupportedError(
        '{} not supported in utensor_cgen'.format(op_type)
      )
    return op_cls(op_info)

  @classmethod
  def register(cls, op_cls):
    cls._operators[op_cls.op_type] = op_cls
    return op_cls

  @classmethod
  def support_op_types(cls):
    """Return the set of all supported ops
    """
    return set(cls._operators.keys())

  @classmethod
  def is_supported(cls, op_type):
    if op_type != 'Placeholder' and op_type not in cls._operators:
      return False
    return True


class _OperatorMeta(type):

  def __new__(mcls, name, bases, attrib):
    attrib['_cache'] = {}
    cls = type.__new__(mcls, name, bases, attrib)
    return cls


class _Operator(with_metaclass(_OperatorMeta), object):

  def __new__(cls, op_info):
    in_dtypes = tuple(t.dtype for t in op_info.input_tensors)
    out_dtypes = tuple(t.dtype for t in op_info.output_tensors)
    type_signature = (in_dtypes, out_dtypes)
    if type_signature not in cls._cache:
      self = object.__new__(cls)
      self.in_dtypes = in_dtypes
      self.out_dtypes = out_dtypes
      self.attributes = deepcopy(op_info.op_attr)
      cls._cache[type_signature] = self
    return cls._cache[type_signature]

  def get_declare_snippet(self, op_var_name, **kwargs):
    raise NotImplementedError(
      'base get_declare_snippet invoked: {}'.format(type(self))
    )

  def get_eval_snippet(self, op_info, op_name, tensor_var_map, **kwargs):
    raise NotImplementedError(
      'base get_eval_snippet invoked: {}'.format(type(self))
    )


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
