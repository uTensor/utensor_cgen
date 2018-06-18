import pytest
from random import shuffle
from utensor_cgen.transformer import (RefCntOptimizer, DropoutTransformer,
                                      BatchNormTransformer, QuantizeTransformer)

@pytest.fixture(scope='function', name='methods')
def pipeline_methods():
    all_methods = [BatchNormTransformer.KWARGS_NAMESCOPE,
                   DropoutTransformer.KWARGS_NAMESCOPE,
                   QuantizeTransformer.KWARGS_NAMESCOPE,
                   RefCntOptimizer.KWARGS_NAMESCOPE]
    shuffle(all_methods)
    return all_methods
