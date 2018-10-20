import pytest
import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow import make_tensor_proto
from utensor_cgen.ir.converter import TensorProtoConverter


@pytest.fixture(scope='session')
def generic_array():
    np_array = np.random.randn(3, 3).astype(np.float32)
    return TensorProtoConverter.__utensor_generic_type__(np_array=np_array)


@pytest.fixture(scope='session')
def tf_qint8_tensor():
    return make_tensor_proto(127*np.random.rand(3, 3),
                             types_pb2.DT_QINT8)

@pytest.fixture(scope='session')
def tf_quint8_tensor():
    return make_tensor_proto(255*np.random.rand(3, 3),
                             types_pb2.DT_QUINT8)