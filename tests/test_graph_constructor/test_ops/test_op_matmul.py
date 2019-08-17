import numpy as np


def test_op_matmul(ugraph, quant_trans):
    tensor_x, = ugraph.add_op(
        op_type='Const',
        name='x',
        value=np.random.rand(3, 5)
    )
    tensor_w, = ugraph.add_op(
        op_type='Const',
        name='w',
        value=np.random.rand(5, 4)
    )
    tensor_z, = ugraph.add_op(
        tensor_x, tensor_w,
        op_type='MatMul',
        name='z'
    )

    assert tensor_z.shape == [3, 4]
    quant_trans.transform(ugraph)
