import numpy as np
import tensorflow as tf
from tensorflow import as_dtype

from utensor_cgen.ir.converter import DataTypeConverter


def test_float32():
    tf_float = as_dtype(np.float32).as_datatype_enum
    np_float = DataTypeConverter.get_generic_value(tf_float)
    assert isinstance(np_float, DataTypeConverter.__utensor_generic_type__)
    assert isinstance(DataTypeConverter.get_tf_value(np_float), 
                      DataTypeConverter.__tfproto_type__)

def test_float64():
    tf_float = as_dtype(np.float64).as_datatype_enum
    np_float = DataTypeConverter.get_generic_value(tf_float)
    assert isinstance(np_float, DataTypeConverter.__utensor_generic_type__)
    assert isinstance(DataTypeConverter.get_tf_value(np_float), 
                      DataTypeConverter.__tfproto_type__)

def test_int32():
    tf_int = as_dtype(np.int32).as_datatype_enum
    np_float = DataTypeConverter.get_generic_value(tf_int)
    assert isinstance(np_float, DataTypeConverter.__utensor_generic_type__)
    assert isinstance(DataTypeConverter.get_tf_value(np_float), 
                      DataTypeConverter.__tfproto_type__)

def test_int64():
    tf_int = as_dtype(np.int64).as_datatype_enum
    np_float = DataTypeConverter.get_generic_value(tf_int)
    assert isinstance(np_float, DataTypeConverter.__utensor_generic_type__)
    assert isinstance(DataTypeConverter.get_tf_value(np_float), 
                      DataTypeConverter.__tfproto_type__)

def test_qint8():
    tf_qint8 = as_dtype(tf.qint8).as_datatype_enum
    np_int8 = DataTypeConverter.get_generic_value(tf_qint8)
    assert np_int8 == np.dtype('int8')
    assert isinstance(np_int8, DataTypeConverter.__utensor_generic_type__)
    assert isinstance(DataTypeConverter.get_tf_value(np_int8),
                      DataTypeConverter.__tfproto_type__)

def test_quint8():
    tf_quint8 = as_dtype(tf.quint8).as_datatype_enum
    np_uint8 = DataTypeConverter.get_generic_value(tf_quint8)
    assert np_uint8 == np.dtype('uint8')
    assert isinstance(np_uint8, DataTypeConverter.__utensor_generic_type__)
    assert isinstance(DataTypeConverter.get_tf_value(np_uint8),
                      DataTypeConverter.__tfproto_type__)
