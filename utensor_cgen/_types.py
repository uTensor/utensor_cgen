# -*- coding:utf8 -*-
from collections import namedtuple

import tensorflow as tf

_TYPE_MAP_VALUE = namedtuple("_TYPE_MAP_VALUE", ["importer_type_str", "tensor_type_str"])

TF_TYPES_MAP = {
  tf.float32: _TYPE_MAP_VALUE(importer_type_str="float", tensor_type_str="float"),
  tf.qint8: _TYPE_MAP_VALUE(importer_type_str="byte", tensor_type_str="uint8_t"),
  tf.int32: _TYPE_MAP_VALUE(importer_type_str="int", tensor_type_str="int"),
  tf.int64: _TYPE_MAP_VALUE(importer_type_str="int", tensor_type_str="int"),
  tf.quint8: _TYPE_MAP_VALUE(importer_type_str="ubyte", tensor_type_str="uint8_t"),
  tf.qint32: _TYPE_MAP_VALUE(importer_type_str="int", tensor_type_str="int")
}
