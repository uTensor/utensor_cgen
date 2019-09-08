import numpy as np


def test_op_min(ugraph, quant_trans):
    with ugraph.begin_construction():
        tensor_x, = ugraph.add_op(
            np.random.rand(3, 5, 9, 2).astype('float32'),
            op_type='Const',
            name='x'
        )
        tensor_out1, = ugraph.add_op(
            tensor_x,
            op_type='Min',
            name='max',
            axis=0,
            keepdims=True,
            is_output=True,
        )
        tensor_out2, = ugraph.add_op(
            tensor_x,
            op_type='Min',
            name='max2',
            axis=1,
            keepdims=True,
            is_output=True,
        )
        tensor_out3, = ugraph.add_op(
            tensor_x,
            op_type='Min',
            name='max3',
            axis=-1,
            keepdims=False,
            is_output=True,
        )
    assert tensor_out1.shape == [1, 5, 9, 2]
    assert tensor_out2.shape == [3, 1, 9, 2]
    assert tensor_out3.shape == [3, 5, 9]
    quant_trans.transform(ugraph)
