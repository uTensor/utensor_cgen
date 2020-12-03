import numpy as np


def test_op_conv2d(ugraph):
    with ugraph.begin_construction():
        tensor_x, = ugraph.add_op(
            'Constant',
            values=np.random.rand(10, 512, 512, 5),
            name='feature_map'
        )
        tensor_w, = ugraph.add_op(
            'Constant',
            values=np.random.rand(32, 32, 5, 10),
            name='filter'
        )
        out, = ugraph.add_op(
            'Conv2dOperator',
            tensor_x,
            tensor_w,
            name='output',
            padding='SAME',
            stride_height=2,
            stride_width=2,
            is_output=True
        )
    assert out.shape == [10, 256, 256, 10]
