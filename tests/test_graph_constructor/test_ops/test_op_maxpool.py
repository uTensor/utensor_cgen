import numpy as np


def test_op_maxpool(ugraph):
    with ugraph.begin_construction():
        tensor_x, = ugraph.add_op(
            'Constant',
            values=np.random.rand(10, 256, 256, 5),
            name='x'
        )
        tensor_out, = ugraph.add_op(
            'MaxPoolOperator',
            tensor_x,
            name='pool',
            ksize_height=32,
            ksize_width=32,
            stride_height=2,
            stride_width=2,
            padding='SAME',
            is_output=True
        )
    assert tensor_out.shape == [10, 128, 128, 5]
