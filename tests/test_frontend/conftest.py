import os

import pytest


@pytest.fixture(name='tflm_mnist_path', scope='session')
def tflm_mnist_path():
    model_path = os.path.join(
        os.path.dirname(__file__),
        'model_files',
        'quant_mnist_cnn.tflite'
    )
    return model_path
