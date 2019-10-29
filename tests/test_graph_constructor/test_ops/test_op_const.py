import numpy as np


def test_op_const(ugraph, quant_trans):
    with ugraph.begin_construction():
        out_tensor, = ugraph.add_op(
            op_type='Const',
            name='ones',
            value=np.ones((3, 3), dtype=np.dtype('float32')),
            is_output=True
        )

    assert out_tensor.shape == [3, 3]
    quant_trans.transform(ugraph)
