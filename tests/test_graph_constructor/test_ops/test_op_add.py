import numpy as np


def test_op_add(ugraph):
    with ugraph.begin_construction():
        tensor_x, = ugraph.add_op(
            op_type='Const',
            name='x',
            value=np.random.rand(1, 3, 5).astype('float32')
        )
        tensor_y, = ugraph.add_op(
            op_type='Const',
            name='y',
            value=np.random.rand(1, 3, 5).astype('float32')
        )
        tensor_z, = ugraph.add_op(
            tensor_x,
            tensor_y,
            op_type='Add',
            name='z',
            is_output=True
        )

    assert tensor_z.shape == [1, 3, 5]
