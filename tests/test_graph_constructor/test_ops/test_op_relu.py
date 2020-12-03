import numpy as np


def test_op_relu(ugraph):
    with ugraph.begin_construction():
        tensor_x, = ugraph.add_op(
            'Constant',
            name='x',
            values=np.random.rand(3, 5)
        )
        out, = ugraph.add_op(
            'ReLUOperator',
            tensor_x,
            name='relu',
            is_output=True
        )
    assert out.shape == [3, 5]
