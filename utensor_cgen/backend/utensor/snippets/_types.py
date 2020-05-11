# -*- coding:utf8 -*-
from collections import namedtuple

import numpy as np

from utensor_cgen.utils import LazyLoader

tf = LazyLoader('tensorflow')


class NumpyTypesMap(object):
  _obj = None
  _NP_TYPES_MAP = {}
  _inited = False

  def __new__(cls):
    if cls._obj is None:
      self = object.__new__(cls)
      cls._obj = self
    return cls._obj

  def __getitem__(self, key):
    cls = type(self)
    cls._init()
    return cls._NP_TYPES_MAP[key]
  
  def __setitem__(self, key, value):
    cls = type(self)
    cls._init()
    cls._NP_TYPES_MAP[key] = value

  def __contains__(self, key):
    cls = type(self)
    cls._init()
    return key in cls._NP_TYPES_MAP
  
  @classmethod
  def _init(cls):
    if not cls._inited:
      _TYPE_MAP_VALUE = namedtuple("_TYPE_MAP_VALUE", ["importer_type_str", "tensor_type_str"])
      cls._NP_TYPES_MAP = {
        np.dtype(tf.float32.as_numpy_dtype): _TYPE_MAP_VALUE(
          importer_type_str="float",
          tensor_type_str="float"
        ),
        np.dtype(tf.qint8.as_numpy_dtype): _TYPE_MAP_VALUE(
          importer_type_str="byte",
          tensor_type_str="uint8_t"
        ),
        np.dtype(tf.int32.as_numpy_dtype): _TYPE_MAP_VALUE(
          importer_type_str="int",
          tensor_type_str="int"
        ),
        np.dtype(tf.int64.as_numpy_dtype): _TYPE_MAP_VALUE(
          importer_type_str="int",
          tensor_type_str="int"
        ),
        np.dtype(tf.quint8.as_numpy_dtype): _TYPE_MAP_VALUE(
          importer_type_str="ubyte",
          tensor_type_str="uint8_t"
        ),
        np.dtype(tf.qint32.as_numpy_dtype): _TYPE_MAP_VALUE(
          importer_type_str="int",
          tensor_type_str="int"
        ),
        np.dtype('uint16'): _TYPE_MAP_VALUE(
          importer_type_str="ushort",
          tensor_type_str="uint16_t"
        ),
        np.dtype('int8'): _TYPE_MAP_VALUE(
          importer_type_str="int8",
          tensor_type_str="int8_t" #"q7_t"
        ),
        np.dtype('uint8'): _TYPE_MAP_VALUE(
          importer_type_str="uint8",
          tensor_type_str="uint8_t"
        ),
      }
      cls._inited = True

NP_TYPES_MAP = NumpyTypesMap()

class Numpy2uTensorTypesMap(object):
  _obj = None
  _NP_TYPES_MAP = {}
  _inited = False

  def __new__(cls):
    if cls._obj is None:
      cls._obj = object.__new__(cls)
    return cls._obj

  def __getitem__(self, key):
    cls = type(self)
    cls._init()
    return cls._NP_TYPES_MAP.get(key, 'undefined')

  def __contains__(self, key):
    cls = type(self)
    cls._init()
    return key in cls._NP_TYPES_MAP
  
  @classmethod
  def _init(cls):
    if not cls._inited:
      cls._NP_TYPES_MAP = {
        np.dtype('int8'): 'i8',
        np.dtype('uint8'): 'u8',
        np.dtype('int16'): 'i16',
        np.dtype('uint16'): 'u16',
        np.dtype('int32'): 'i32',
        np.dtype('uint32'): 'u32',
        np.dtype('float32'): 'flt',
      }
      cls._inited = True

UTENSOR_TYPES_MAP = Numpy2uTensorTypesMap()
