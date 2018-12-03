r"""This module defines the interface of a converter.
A converter is responsible for converting tensorflow/pytorch types to 
generic python type
"""
import sys
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from functools import wraps

import attr
import numpy as np
from attr import validators
from tensorflow import DType as _DType
from tensorflow import as_dtype as _tf_as_dtype
from tensorflow import make_ndarray, make_tensor_proto
from tensorflow.core.framework.attr_value_pb2 import AttrValue as _AttrValue
from tensorflow.core.framework.attr_value_pb2 import \
    NameAttrList as _NameAttrList
from tensorflow.core.framework.tensor_pb2 import TensorProto as _TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import \
    TensorShapeProto as _TensorShapeProto
from tensorflow.core.framework.types_pb2 import DataType as _DataType
from tensorflow.python.framework import tensor_shape

from .utils import is_list_of

__all__ = ['ConverterFactory']

if sys.version_info.major > 2:
  long = int
  unicode = str

class GenericConverter(object):
  """Convert value to utensor generic type
  """
  __metaclass__ = ABCMeta
  __utensor_generic_type__ = None

  @classmethod
  @abstractmethod
  def get_generic_value(cls, value):
      raise NotImplementedError('')

class TFConverterMixin(object):
  """Convert value to tf protobuf types
  """
  __tfproto_type__ = None

  @classmethod
  @abstractmethod
  def get_tf_value(cls, value):
    raise NotImplementedError('')

class GenericTensorConverterMixin(GenericConverter):
  @attr.s
  class GenericType(object):
    np_array = attr.ib(validator=validators.instance_of(np.ndarray))
    dtype = attr.ib(default=None)
    
    def __attrs_post_init__(self):
      if self.dtype is None:
        self.dtype = self.np_array.dtype
  __utensor_generic_type__ = GenericType

class GenericDataTypeConverterMixin(GenericConverter):
  __utensor_generic_type__ = np.dtype

class GenericTensorShapeMixin(GenericConverter):
  @attr.s
  class GenericType(object):
    list_view = attr.ib()
    
    @list_view.validator
    def check(self, attrib, a_list):
      if a_list is None:
        # unknown shape
        return
      if not isinstance(a_list, list):
        raise ValueError('list_view should be a list')
      # check the values in the list
      is_valid = True
      for v in a_list:
        if isinstance(v, (int, type(None))):
          continue
        is_valid = False
      if not is_valid:
        raise ValueError("Invalid value type for %s" % a_list)
  __utensor_generic_type__ = GenericType

# helpers
def _check_tf_type(conv_func):
  @wraps(conv_func)
  def wrap(cls, value):
    assert isinstance(value, cls.__tfproto_type__), \
      "Expecting %s, get %s" % (cls.__tfproto_type__, type(value))
    return conv_func(cls, value)
  return wrap

def _check_generic_type(conv_func):
  @wraps(conv_func)
  def wrap(cls, value):
    assert isinstance(value, cls.__utensor_generic_type__), \
      "Expecting %s, get %s" % (cls.__utensor_generic_type__, type(value))
    return conv_func(cls, value)
  return wrap

class BuiltinConverter(GenericConverter, TFConverterMixin):
  """Identity converter for buildin types
  """
  __tfproto_type__ = (bytes, int, long, bool, float, str, unicode)
  __utensor_generic_type__ = __tfproto_type__

  @classmethod
  @_check_tf_type
  def get_generic_value(cls, value):
    return value

  @classmethod
  @_check_generic_type
  def get_tf_value(cls, value):
    return value

class ConverterFactory(object):
  _BUILTIN_MAP = dict((t, BuiltinConverter) for t in BuiltinConverter.__tfproto_type__)
  _TF2GENERIC_MAP = {}
  _TF2GENERIC_MAP.update(_BUILTIN_MAP)

  _GENERIC2TF_MAP = {}
  _GENERIC2TF_MAP.update(_BUILTIN_MAP)

  @classmethod
  def register(cls, converter_cls):
    if not issubclass(converter_cls, GenericConverter) or \
      not issubclass(converter_cls, TFConverterMixin):
      raise ValueError(
        ('converter has to be subclass of both GenericConverter and TFConverterMixin'
         ': %s' % converter_cls)
      )
    if converter_cls.__utensor_generic_type__ is None:
      raise ValueError('__utensor_generic_type__ cannot be None: %s' % converter_cls)
    if converter_cls.__tfproto_type__ is None:
      raise ValueError('__tfproto_type__ cannot be None: %s' % converter_cls)
    cls._TF2GENERIC_MAP[converter_cls.__tfproto_type__] = converter_cls
    cls._GENERIC2TF_MAP[converter_cls.__utensor_generic_type__] = converter_cls
    return converter_cls

  @classmethod
  def get_generic_value(cls, tf_value):
    value_type = type(tf_value)
    if value_type in cls._GENERIC2TF_MAP:
      # already generic type
      return tf_value
    cvt = cls._TF2GENERIC_MAP.get(value_type, None)
    if not cvt:
      raise ValueError('Unknown tf value type: %s' % value_type)
    return cvt.get_generic_value(tf_value)
  
  @classmethod
  def get_tf_value(cls, generic):
    value_type = type(generic)
    if value_type in cls._TF2GENERIC_MAP:
      # already tf type
      return generic
    cvt = cls._GENERIC2TF_MAP.get(value_type, None)
    if not cvt:
      raise ValueError('Unknown generic type: %s' % value_type)
    return cvt.get_tf_value(generic)
  
  @classmethod
  def all_supported_tf_types(cls):
    return cls._TF2GENERIC_MAP.keys()
  
  @classmethod
  def all_generic_types(cls):
    return cls._GENERIC2TF_MAP.keys()

  @classmethod
  def TF2GENERIC_MAP(cls):
    type_map = {}
    for converter in cls._TF2GENERIC_MAP.values():
      type_map[converter.__tfproto_type__] = converter.__utensor_generic_type__
    return type_map

# converters
@ConverterFactory.register
class TensorProtoConverter(GenericTensorConverterMixin, TFConverterMixin):
  __tfproto_type__ = _TensorProto

  @classmethod
  @_check_generic_type
  def get_tf_value(cls, value):
    return make_tensor_proto(value.np_array, dtype=value.dtype)
  
  @classmethod
  @_check_tf_type
  def get_generic_value(cls, value):
    """quantized numpy array is mapped to np.uint8 typed

    FIXME: I'm not sure if it's a good idea
    """
    np_array = make_ndarray(value)
    dtype = np_array.dtype
    if dtype.fields is None:
      pass
    elif dtype[0] in [np.uint8, np.int8]:
      np_array = np_array.astype(dtype[0])
    else:
      raise ValueError('Unsupported numpy dtype: %s' % dtype)
    return cls.__utensor_generic_type__(np_array=np_array,
                                        dtype=dtype)

@ConverterFactory.register
class DataTypeConverter(GenericDataTypeConverterMixin, TFConverterMixin):
  __tfproto_type__ = int # _DataType is an enum type
  
  @classmethod
  @_check_generic_type
  def get_tf_value(cls, value):
    return _tf_as_dtype(value).as_datatype_enum
  
  @classmethod
  @_check_tf_type
  def get_generic_value(cls, value):
    dtype = _DType(value)
    np_dtype = np.dtype(dtype.as_numpy_dtype)
    return cls._handle_qtype(np_dtype)
  
  @classmethod
  def _handle_qtype(cls, dtype):
    """Mapping qantized dtype to np.uint8

    FIXME: I'm not sure if it's a good idea
    """
    if dtype.fields is None:
      return dtype
    if dtype[0] in [np.uint8]:
      return np.dtype('uint8')
    elif dtype[0] in [np.int8]:
      return np.dtype('int8')
    else:
      raise TypeError('Unsupport numpy dtype: %s' % dtype)

@ConverterFactory.register
class TensorShapeConverter(GenericTensorShapeMixin, TFConverterMixin):
  __tfproto_type__ = _TensorShapeProto

  @classmethod
  @_check_generic_type
  def get_tf_value(cls, value):
    return tensor_shape.TensorShape(value.list_view).as_proto()

  @classmethod
  @_check_tf_type
  def get_generic_value(cls, value):
    try:
      list_view = tensor_shape.TensorShape(value).as_list()
    except ValueError:
      # unknown shape
      list_view = None
    return cls.GenericType(list_view=list_view)

@ConverterFactory.register
class AttrValueConverter(GenericConverter, TFConverterMixin):
  __tfproto_type__ = _AttrValue

  @attr.s
  class GenericType(object):
    value_name = attr.ib(validator=validators.instance_of((str, unicode)))
    value = attr.ib()
    @value.validator
    def check(self, attrib, value):
      all_types = (bytes, int, float, bool, str, long, unicode,
                   DataTypeConverter.__utensor_generic_type__,
                   TensorShapeConverter.__utensor_generic_type__,
                   TensorProtoConverter.__utensor_generic_type__,
                   AttrListValueConverter.__utensor_generic_type__,
                   NameAttrListConverter.__utensor_generic_type__)
      if not isinstance(value, all_types):
        raise ValueError("Expecting %s, get %s" % (all_types, type(value)))

  __utensor_generic_type__ = GenericType

  @classmethod
  @_check_generic_type
  def get_tf_value(cls, generic):
    value = ConverterFactory.get_tf_value(generic.value)
    return cls.__tfproto_type__(**{generic.value_name : value})
  
  @classmethod
  @_check_tf_type
  def get_generic_value(cls, tf_value):
    value_name = tf_value.WhichOneof('value')
    value = getattr(tf_value, value_name)
    if isinstance(value, np.ndarray):
      value = self._handle_quant_array(value)
    return cls.__utensor_generic_type__(value_name=value_name,
                                        value=ConverterFactory.get_generic_value(value))

@ConverterFactory.register
class NameAttrListConverter(GenericConverter, TFConverterMixin):
  __tfproto_type__ = _NameAttrList

  @attr.s
  class GenericType(object):
    name = attr.ib(validator=validators.instance_of((str, unicode)))
    attr_map = attr.ib()
    @attr_map.validator
    def check(self, attrib, value):
      is_valid = True
      if not isinstance(value, dict):
        is_valid = False
      else:
        for k, v in value.items():
          if (not isinstance(k, (str, unicode)) or 
              not isinstance(v, AttrValueConverter.__utensor_generic_type__)):
            is_valid = False
      if not is_valid:
        raise ValueError(("Expecting a dict with key of str (or unicode), value of %s, get %s" 
                          % (AttrValueConverter.__utensor_generic_type__, value)))
  __utensor_generic_type__ = GenericType

  @classmethod
  @_check_generic_type
  def get_tf_value(cls, generic):
    kwargs = {
      'name': generic.name,
      'attr': dict((k, ConverterFactory.get_tf_value(v))
                   for k, v in generic.attr_map.items())
    }
    return cls.__tfproto_type__(**kwargs)

  @classmethod
  @_check_tf_type
  def get_generic_value(cls, tf_value):
    name = tf_value.name
    attr_map = {}
    for name, attr in tf_value.attr.items():
      attr_map[name] = AttrValueConverter.get_generic_value(attr)
    return cls.__utensor_generic_type__(name=name, attr_map=attr_map)

@ConverterFactory.register
class AttrListValueConverter(GenericConverter, TFConverterMixin):
  __tfproto_type__ = _AttrValue.ListValue

  @attr.s
  class GenericType(object):
    bytes_value = attr.ib(factory=list, validator=is_list_of(bytes))
    ints_value = attr.ib(factory=list, 
                         type=lambda l: [int(e) for e in l])
    floats_value = attr.ib(factory=list, validator=is_list_of(float))
    bools_value = attr.ib(factory=list, validator=is_list_of(bool))
    data_types_value = attr.ib(factory=list, validator=is_list_of(DataTypeConverter.__utensor_generic_type__))
    tensorshape_protos_value = attr.ib(factory=list, validator=is_list_of(TensorShapeConverter.__utensor_generic_type__))
    tensor_protos_value = attr.ib(factory=list, validator=is_list_of(TensorProtoConverter.__utensor_generic_type__))
    name_attrs_value = attr.ib(factory=list, 
                               validator=is_list_of(NameAttrListConverter.__utensor_generic_type__))
    
  __utensor_generic_type__ = GenericType
  
  @classmethod
  @_check_generic_type
  def get_tf_value(cls, generic):
    kwargs = {
      's': generic.bytes_value,
      'i': generic.ints_value,
      'f': generic.floats_value,
      'b': generic.bools_value,
      'type': [ConverterFactory.get_tf_value(v) for v in generic.data_types_value],
      'shape': [ConverterFactory.get_tf_value(v) for v in generic.tensorshape_protos_value],
      'tensor': [ConverterFactory.get_tf_value(v) for v in generic.tensor_protos_value],
      'func': [ConverterFactory.get_tf_value(v) for v in generic.name_attrs_value]
    }
    return cls.__tfproto_type__(**kwargs)

  @classmethod
  @_check_tf_type
  def get_generic_value(cls, tf_value):
    kwargs = {
      'bytes_value': list(tf_value.s),
      'ints_value': list(tf_value.i),
      'floats_value': list(tf_value.f),
      'bools_value': list(tf_value.b),
      'data_types_value': [ConverterFactory.get_generic_value(v) for v in tf_value.type],
      'tensorshape_protos_value': [ConverterFactory.get_generic_value(v) for v in tf_value.shape],
      'tensor_protos_value': [ConverterFactory.get_generic_value(v) for v in tf_value.tensor],
      'name_attrs_value': [ConverterFactory.get_generic_value(v) for v in tf_value.func]
    }
    return cls.__utensor_generic_type__(**kwargs)
