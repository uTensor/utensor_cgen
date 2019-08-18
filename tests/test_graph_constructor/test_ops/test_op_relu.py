import numpy as np


def test_op_relu(ugraph, quant_trans):
    with ugraph.begin_construction():
        tensor_x, = ugraph.add_op(
            op_type='Const',
            name='x',
            value=np.random.rand(3, 5)
        )
        out, = ugraph.add_op(
            tensor_x,
            op_type='Relu',
            name='relu',
            is_output=True
        )
    assert out.shape == [3, 5]
    quant_trans.transform(ugraph)
