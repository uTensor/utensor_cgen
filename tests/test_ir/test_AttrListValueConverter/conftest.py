from tensorflow.core.framework.attr_value_pb2 import AttrValue
import pytest


@pytest.fixture(scope='session')
def int_list():
    return AttrValue.ListValue(i=[1, 2, 3])

@pytest.fixture(scope='session')
def bool_list():
    return AttrValue.ListValue(b=[True, False])
