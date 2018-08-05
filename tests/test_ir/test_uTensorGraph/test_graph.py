from copy import deepcopy

import numpy as np
import tensorflow as tf

from utensor_cgen.ir import OperationInfo, uTensorGraph
from utensor_cgen.ir.converter import TensorProtoConverter


def test_ugraph_topo_order(graph_tuple):
    graph_def, output_nodes = graph_tuple
    ugraph = uTensorGraph(graph_def,
                          output_nodes=output_nodes)
    first_out, second_out = output_nodes
    meet_first = False
    for node_name in ugraph.topo_order:
        if node_name == first_out:
            meet_first = True
        if node_name == second_out:
            assert meet_first

def test_ugraph_copy(graph_tuple):
    graph_def, output_nodes = graph_tuple
    ugraph_1 = uTensorGraph(graph_def,
                            output_nodes=output_nodes)
    ugraph_2 = deepcopy(ugraph_1)
    assert ugraph_1 is not ugraph_2
    assert ugraph_1.graph_def == ugraph_2.graph_def

def test_op_info():
    np_array = np.array([1, 2, 3], dtype=np.float32)
    t_proto = tf.make_tensor_proto(np_array, dtype=np.float32)
    op_info = OperationInfo(name='testing_op',
                            input_tensors=[],
                            output_tensors=[],
                            op_type='no_op',
                            backend='tensorflow',
                            op_attr={
                                '_to_skip': [1, 2, 3],
                                '_skip_this_too': None,
                                'tensor_no_skip': t_proto
                            })
    assert op_info.op_attr.get('_to_skip', None) == [1, 2, 3]
    assert op_info.op_attr.get('_skip_this_too') is None
    generic_tensor = op_info.op_attr.get('tensor_no_skip')
    assert isinstance(generic_tensor,
                      TensorProtoConverter.__utensor_generic_type__)
    assert (generic_tensor.np_array == np_array).all()

def test_in_out_nodes(graph_tuple):
    ugraph = uTensorGraph(*graph_tuple)    
    x3 = ugraph.ops_info['x3']
    assert x3.ugraph is ugraph
    assert len(x3.input_nodes) == len(set([op.name for op in x3.input_nodes]))
    assert all([str(op.name) in ['x2', 'bias2'] for op in x3.input_nodes])
    assert x3.output_nodes == []

    x2 = ugraph.ops_info['x2']
    assert [str(op.name) for op in x2.output_nodes] == ['x3']
