from copy import deepcopy
from typing import Hashable

from six import with_metaclass

from utensor_cgen.logger import logger
from utensor_cgen.utils import MUST_OVERWRITE, must_return_type

__all__ = ["OperatorFactory"]


class OperatorFactory(object):

  _operators = {}
  _warned_missing_ops = set()

  @classmethod
  def get_opertor(cls, op_info):
    op_type = op_info.op_type
    codegen_namespaces = op_info.code_gen_attributes.get('namespaces', tuple())
    op_cls = cls._operators.get((codegen_namespaces, op_type))
    if op_cls is None:
      missing_op_cls = cls._operators['_MissingOperator']
      if op_info.op_type not in cls._warned_missing_ops:
        op_full_name = '::'.join(
          ["uTensor"] + \
          list(op_info.code_gen_attributes.get('namespaces', [])) + \
          [op_info.op_type]
        )
        logger.warning(
          '{} is missing, no code will be generated for it'.format(op_full_name)
        )
        cls._warned_missing_ops.add(op_info.op_type)
      return missing_op_cls(op_info)
    return op_cls(op_info)

  @classmethod
  def register(cls, op_cls):
    cls._operators[
      (op_cls.namespaces, op_cls.op_type)
    ] = op_cls
    return op_cls

  @classmethod
  def support_op_types(cls):
    """Return the set of all supported ops
    """
    return set([
      "::".join(list(namespaces) + [op_type])
      for namespaces, op_type in cls._operators.keys()
    ])

  @classmethod
  def is_supported(cls, op_type):
    if op_type != "Placeholder" and op_type not in cls._operators:
      return False
    return True


class _OperatorMeta(type):
  def __new__(mcls, name, bases, attrib):
    attrib["_cache"] = {}
    cls = type.__new__(mcls, name, bases, attrib)
    for key in ["get_type_signature", "get_constructor_parameters"]:
      func = getattr(cls, key)
      if func is None:
        continue
      if not must_return_type.return_type_is_ensured(func):
        setattr(cls, key, must_return_type(Hashable)(func))
      elif not issubclass(must_return_type.get_expect_type(func), Hashable):
        raise RuntimeError(
          "{}.{} must be ensured to return {}".format(name, key, Hashable)
        )
    return cls


class _Operator(with_metaclass(_OperatorMeta), object):
  namespaces = tuple()
  op_type = MUST_OVERWRITE

  def __new__(cls, op_info):
    if cls.op_type is MUST_OVERWRITE:
      raise ValueError('op_type must be overwritten: {}'.format(cls))

    in_dtypes, out_dtypes = type_signature = cls.get_type_signature(op_info)
    construct_params = cls.get_constructor_parameters(op_info)
    full_signature = (cls.namespaces, type_signature, construct_params)
    if full_signature not in cls._cache:
      self = object.__new__(cls)
      self.in_dtypes = in_dtypes
      self.out_dtypes = out_dtypes
      self.construct_params = construct_params
      self.op_type = op_info.op_type
      cls._cache[full_signature] = self
    return cls._cache[full_signature]

  @classmethod
  @must_return_type(Hashable)
  def get_type_signature(cls, op_info):
    """Type signature for the operator template class

    Parameter
    ---------
    - `op_info`: an `OperatorInfo` object

    Return
    ------
    - tuple

    Note
    ----
    For example, `MyOperator<float, double>` is an operator
    that takes float as input type and double as outpupt type,
    then you should return a hashable object, normally a tuple,
    which is a unique identifier of the operator.

    Note that, all operator are implement as singlton. That is, any
    operator with the same type signature and same constructor parameters
    (see below) will point to the same object in memory.
    """
    in_dtypes = tuple(t.dtype for t in op_info.input_tensors)
    out_dtypes = tuple(t.dtype for t in op_info.output_tensors)
    return (in_dtypes, out_dtypes)

  @classmethod
  @must_return_type(Hashable)
  def get_constructor_parameters(cls, op_info):
    """Parameters for the operator constructor

    Parameter
    ---------
    - `op_info`: an `OperatorInfo` object

    Return
    ------
    - tuple

    Note
    ----
    if we need to `MyConstructor` as `MyConstructor<float>(1, 2, 3)`, we
    need to return a tuple, `(1, 2, 3)`, in this function.

    For the sake of memory efficiency, opertor in uTensor runtime will only
    instantiate once in the snippet. The tuple returned by this method together
    with the tuple returned by `get_type_signature` will serve as unique identifier
    of the operator
    """
    return tuple()

  def get_declare_snippet(self, op_var_name, with_const_params=True, **kwargs):
    """Snippet for delaring the operator

    Parameters
    ----------
    - `op_var_name` (str): the variable name for the operator
    - `with_const_params` (bool): whether to emit constructor parameters in the snippet

    Return
    ------
    an object with `render` method which takes no arguments and return a string, normally
    it's a `DeclareOpSnippet` object.
    """
    raise NotImplementedError(
      "base get_declare_snippet invoked: {}".format(type(self))
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map, **kwargs):
    """Snippet for evaluate the operator

    Parameters
    ----------
    - `op_var_name` (str): the variable name for the operator
    - `tensor_var_map` (dict): a dictionary which maps tensor name in the graph to 
      its variable name
    
    Return
    ------
    an object with `render` method which takes no arguments and return a string, normally
    it's an object of subclass of OpEvalSnippet
    """
    raise NotImplementedError(
      "base get_eval_snippet invoked: {}".format(type(self))
    )
  
  def get_construct_snippet(self, op_var_name):
    """Constructor Snippet of the operator

    Parameters
    ----------
    - `op_var_name` (str): the variable name of the operator

    Return
    ------
    an object with `render` method which takes no arguments and return a string, normally
    it's an object of OpConstructSnippet
    """
    raise NotImplementedError(
      "base get_construct_snippet invoked: {}".format(type(self))
    )
