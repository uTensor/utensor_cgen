import pytest
import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow.contrib.util import make_tensor_proto


@pytest.fixture(scope='session')
def np_array():
    return np.random.randn(3, 3)

@pytest.fixture(scope='session')
def tf_qint8_tensor():
    return make_tensor_proto(127*np.random.rand(3, 3),
                             types_pb2.DT_QINT8)

@pytest.fixture(scope='session')
def tf_quint8_tensor():
    return make_tensor_proto(255*np.random.rand(3, 3),
                             types_pb2.DT_QUINT8)