# -*- coding:utf8 -*-
from collections import namedtuple

import numpy as np
import tensorflow as tf

_TYPE_MAP_VALUE = namedtuple("_TYPE_MAP_VALUE", ["importer_type_str", "tensor_type_str"])

NP_TYPES_MAP = {
  np.dtype(tf.float32.as_numpy_dtype): _TYPE_MAP_VALUE(importer_type_str="float",
                                                       tensor_type_str="float"),
  np.dtype(tf.qint8.as_numpy_dtype): _TYPE_MAP_VALUE(importer_type_str="byte", 
                                                     tensor_type_str="uint8_t"),
  np.dtype(tf.int32.as_numpy_dtype): _TYPE_MAP_VALUE(importer_type_str="int", 
                                                     tensor_type_str="int"),
  np.dtype(tf.int64.as_numpy_dtype): _TYPE_MAP_VALUE(importer_type_str="int", 
                                                     tensor_type_str="int"),
  np.dtype(tf.quint8.as_numpy_dtype): _TYPE_MAP_VALUE(importer_type_str="ubyte",
                                                      tensor_type_str="uint8_t"),
  np.dtype(tf.qint32.as_numpy_dtype): _TYPE_MAP_VALUE(importer_type_str="int", 
                                                      tensor_type_str="int"),
  np.dtype('uint16'): _TYPE_MAP_VALUE(importer_type_str="ushort",
                                                      tensor_type_str="uint16_t"),
  np.dtype('int8'): _TYPE_MAP_VALUE(importer_type_str="int8",
                                                      tensor_type_str="q7_t"),
}
del _TYPE_MAP_VALUE
