import numpy as np


def test_op_conv2d(ugraph, quant_trans):
    with ugraph.begin_construction():
        tensor_x, = ugraph.add_op(
            np.random.rand(10, 512, 512, 5),
            op_type='Const',
            name='feature_map'
        )
        tensor_w, = ugraph.add_op(
            np.random.rand(32, 32, 5, 10),
            op_type='Const',
            name='filter'
        )
        out, = ugraph.add_op(
            tensor_x,
            tensor_w,
            name='output',
            op_type='Conv2D',
            padding='SAME',
            stride_height=2,
            stride_width=2,
            is_output=True
        )
    assert out.shape == [10, 256, 256, 10]
    quant_trans.transform(ugraph)
