import pytest
from random import shuffle
from utensor_cgen.transformer import (RefCntOptimizer, DropoutTransformer,
                                      BatchNormTransformer, QuantizeTransformer)

@pytest.fixture(scope='function', name='methods')
def pipeline_methods():
    all_methods = [BatchNormTransformer.METHOD_NAME,
                   DropoutTransformer.METHOD_NAME,
                   QuantizeTransformer.METHOD_NAME,
                   RefCntOptimizer.METHOD_NAME]
    shuffle(all_methods)
    return all_methods
