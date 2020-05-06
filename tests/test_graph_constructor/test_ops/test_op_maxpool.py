import numpy as np


def test_op_maxpool(ugraph):
    with ugraph.begin_construction():
        tensor_x, = ugraph.add_op(
            np.random.rand(10, 256, 256, 5),
            op_type='Const',
            name='x'
        )
        tensor_out, = ugraph.add_op(
            tensor_x,
            op_type='MaxPool',
            name='pool',
            ksize_height=32,
            ksize_width=32,
            stride_height=2,
            stride_width=2,
            padding='SAME',
            is_output=True
        )
    assert tensor_out.shape == [10, 128, 128, 5]
