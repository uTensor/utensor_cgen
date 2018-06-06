import pytest
import numpy as np


@pytest.fixture(scope='session')
def np_array():
    return np.random.randn(3, 3)
