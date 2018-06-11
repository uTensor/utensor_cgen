from utensor_cgen.ir.converter import TensorProtoConverter
import numpy as np

def test_np_array(np_array):
    tf_value = TensorProtoConverter.get_tf_value(np_array)
    assert isinstance(tf_value, TensorProtoConverter.__tfproto_type__)
    generic = TensorProtoConverter.get_generic_value(tf_value)
    assert isinstance(generic, type(np_array))
    assert (generic == np_array).all()

def test_tf_tensor_qint8(tf_qint8_tensor):
    np_array = TensorProtoConverter.get_generic_value(tf_qint8_tensor)
    assert np_array.dtype == np.dtype('int8')
    tf_value = TensorProtoConverter.get_tf_value(np_array)
    assert tf_value.tensor_content == tf_qint8_tensor.tensor_content
    assert tf_value.tensor_shape == tf_qint8_tensor.tensor_shape


def test_tf_tensor_quint8(tf_quint8_tensor):
    np_array = TensorProtoConverter.get_generic_value(tf_quint8_tensor)
    assert np_array.dtype == np.dtype('uint8')
    tf_value = TensorProtoConverter.get_tf_value(np_array)
    assert tf_value.tensor_content == tf_quint8_tensor.tensor_content
    assert tf_value.tensor_shape == tf_quint8_tensor.tensor_shape
