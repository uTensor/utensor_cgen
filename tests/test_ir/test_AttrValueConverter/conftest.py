import pytest
from tensorflow import AttrValue, NameAttrList


@pytest.fixture(scope='session')
def int_attr_value():
    return AttrValue(i=3333)

@pytest.fixture(scope='session')
def bytes_attr_value():
    return AttrValue(s=b'bytes')

@pytest.fixture(scope='session')
def float_attr_value():
    return AttrValue(f=3.14159)

@pytest.fixture(scope='session')
def bool_attr_value():
    return AttrValue(b=True)

@pytest.fixture(scope='session')
def func_attr_value():
    func = NameAttrList(name='test_attr_list_name',
                        attr={
                            'int': AttrValue(i=3333),
                            'bool': AttrValue(b=True)
                        })
    return AttrValue(func=func)

@pytest.fixture(scope='session')
def placeholder_attr_value():
    return AttrValue(placeholder='placeholder')
