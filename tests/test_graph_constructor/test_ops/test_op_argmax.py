import numpy as np
import pytest


@pytest.mark.deprecated
def test_op_argmax(ugraph):
    with ugraph.begin_construction():
        tensor_logits, = ugraph.add_op(
            np.random.rand(3, 5, 7).astype('float32'),
            op_type='Const',
            name='logits'
        )
        tensor_out1, = ugraph.add_op(
            tensor_logits,
            op_type='ArgMax',
            name='argmax',
            axis=0,
            dtype=np.dtype('int32'),
            is_output=True,
        )
        tensor_out2, = ugraph.add_op(
            tensor_logits,
            op_type='ArgMax',
            name='argmax2',
            axis=-1,
            dtype=np.dtype('int64'),
            is_output=True,
        )
    assert tensor_out1.shape == [5, 7]
    assert tensor_out1.dtype == np.dtype('int32')
    assert tensor_out2.shape == [3, 5]
    assert tensor_out2.dtype == np.dtype('int64')
