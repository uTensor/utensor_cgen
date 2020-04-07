from copy import deepcopy

from six import with_metaclass

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
    if not issubclass(op_cls, _OperatorBase):
      raise TypeError(
        'expecting subclass of _OperatorBase, get {}'.format(op_cls)
      )
    if op_cls is _OperatorBase:
      raise ValueError(
        'cannot register abstract op_cls: {}'.format(_OperatorBase)
      )
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


class _OperatorBase(with_metaclass(_OperatorMeta), object):
  def __new__(cls, op_info):
    if cls is _OperatorBase:
      raise RuntimeError('_OperatorBase should not be instantiated')
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
