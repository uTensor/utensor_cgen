from utensor_cgen.ir.converter import TensorProtoConverter


def test_np_array(np_array):
    tf_value = TensorProtoConverter.get_tf_value(np_array)
    assert isinstance(tf_value, TensorProtoConverter.__tfproto_type__)
    generic = TensorProtoConverter.get_generic_value(tf_value)
    assert isinstance(generic, type(np_array))
    assert (generic == np_array).all()
