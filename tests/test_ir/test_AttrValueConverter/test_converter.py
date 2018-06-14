from utensor_cgen.ir.converter import AttrValueConverter


def test_int_value(int_attr_value):
    generic = AttrValueConverter.get_generic_value(int_attr_value)
    assert isinstance(generic, AttrValueConverter.__utensor_generic_type__)
    tf_proto = AttrValueConverter.get_tf_value(generic)
    assert isinstance(tf_proto, AttrValueConverter.__tfproto_type__)

def test_bytes_value(bytes_attr_value):
    generic = AttrValueConverter.get_generic_value(bytes_attr_value)
    assert isinstance(generic, AttrValueConverter.__utensor_generic_type__)
    tf_proto = AttrValueConverter.get_tf_value(generic)
    assert isinstance(tf_proto, AttrValueConverter.__tfproto_type__)

def test_float_value(float_attr_value):
    generic = AttrValueConverter.get_generic_value(float_attr_value)
    assert isinstance(generic, AttrValueConverter.__utensor_generic_type__)
    tf_proto = AttrValueConverter.get_tf_value(generic)
    assert isinstance(tf_proto, AttrValueConverter.__tfproto_type__)

def test_bool_value(bool_attr_value):
    generic = AttrValueConverter.get_generic_value(bool_attr_value)
    assert isinstance(generic, AttrValueConverter.__utensor_generic_type__)
    tf_proto = AttrValueConverter.get_tf_value(generic)
    assert isinstance(tf_proto, AttrValueConverter.__tfproto_type__)

def test_func_value(func_attr_value):
    generic = AttrValueConverter.get_generic_value(func_attr_value)
    assert isinstance(generic, AttrValueConverter.__utensor_generic_type__)
    tf_proto = AttrValueConverter.get_tf_value(generic)
    assert isinstance(tf_proto, AttrValueConverter.__tfproto_type__)

def test_placeholder_value(placeholder_attr_value):
    generic = AttrValueConverter.get_generic_value(placeholder_attr_value)
    assert isinstance(generic, AttrValueConverter.__utensor_generic_type__)
    tf_proto = AttrValueConverter.get_tf_value(generic)
    assert isinstance(tf_proto, AttrValueConverter.__tfproto_type__)
