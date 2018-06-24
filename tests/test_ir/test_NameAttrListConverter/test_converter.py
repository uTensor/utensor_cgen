from utensor_cgen.ir.converter import NameAttrListConverter


def test_name_attr_list_converter(name_attr_list):
    generic = NameAttrListConverter.get_generic_value(name_attr_list)
    assert isinstance(generic, NameAttrListConverter.__utensor_generic_type__)
    tf_proto = NameAttrListConverter.get_tf_value(generic)
    assert isinstance(tf_proto, NameAttrListConverter.__tfproto_type__)
