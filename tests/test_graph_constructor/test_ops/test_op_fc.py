import numpy as np


def test_op_fc(ugraph):
    with ugraph.begin_construction():
        tensor_x, = ugraph.add_op(
            'Constant',
            name='x',
            values=np.random.rand(3, 5)
        )
        tensor_w, = ugraph.add_op(
            'Constant',
            name='w',
            values=np.random.rand(5, 4)
        )
        tensor_z, = ugraph.add_op(
            'FullyConnectedOperator',
            tensor_x, tensor_w,
            name='z',
            is_output=True
        )

    assert tensor_z.shape == [3, 4]
