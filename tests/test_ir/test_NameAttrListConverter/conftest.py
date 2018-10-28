import pytest
from tensorflow.core.framework.attr_value_pb2 import AttrValue, NameAttrList


@pytest.fixture(scope='session')
def name_attr_list():
    attr = {
        'float': AttrValue(f=3.14159),
        'list': AttrValue(list=AttrValue.ListValue(b=[True, False, True]))
    }
    return NameAttrList(name='test_name_attr_list', attr=attr)
