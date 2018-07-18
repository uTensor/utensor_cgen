import six

from utensor_cgen.ir.converter import AttrListValueConverter


def test_int_list(int_list):
    generic_list = AttrListValueConverter.get_generic_value(int_list)
    assert isinstance(generic_list, AttrListValueConverter.__utensor_generic_type__)
    assert all([isinstance(i, six.integer_types) for i in generic_list.ints_value])
    tf_list = AttrListValueConverter.get_tf_value(generic_list)
    assert isinstance(tf_list, AttrListValueConverter.__tfproto_type__)
    assert tf_list == int_list

def test_bool_list(bool_list):
    generic_list = AttrListValueConverter.get_generic_value(bool_list)
    assert isinstance(generic_list, AttrListValueConverter.__utensor_generic_type__)
    assert all([isinstance(b, bool) for b in generic_list.ints_value])
    tf_list = AttrListValueConverter.get_tf_value(generic_list)
    assert isinstance(tf_list, AttrListValueConverter.__tfproto_type__)
    assert tf_list == bool_list
