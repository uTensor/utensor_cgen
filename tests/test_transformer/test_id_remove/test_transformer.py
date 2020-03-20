from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.transformer.optimizer import IdOpRemoveOptimizer

def test_id_rm_transform_1(id_graph_def_1):
    ugraph = GraphDefParser.parse(id_graph_def_1, output_nodes=['z'])
    optimizer = IdOpRemoveOptimizer()
    new_ugraph = optimizer.transform(ugraph)
    for op in new_ugraph.ops_info.values():
        assert op.op_type != 'Identity'
    op_z = new_ugraph.ops_info['z']
    in_op_names = set([op.name for op in op_z.input_nodes])
    assert set(['x', 'y']) == in_op_names

def test_id_rm_transform_2(id_graph_def_2):
    ugraph = GraphDefParser.parse(id_graph_def_2, output_nodes=['z'])
    optimizer = IdOpRemoveOptimizer()
    new_ugraph = optimizer.transform(ugraph)
    for op in new_ugraph.ops_info.values():
        assert op.op_type != 'Identity'
    op_z = new_ugraph.ops_info['z']
    in_op_names = set([op.name for op in op_z.input_nodes])
    assert set(['w', 'y']) == in_op_names