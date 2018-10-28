import tensorflow as tf

from utensor_cgen.ir import uTensorGraph
from utensor_cgen.transformer.ns_transformer import DropoutTransformer


def test_dropout_trans(droput_graph_tuple):
    (graph_def,
     (keep_prob_name, dropout_output_name),
     output_nodes) = droput_graph_tuple
    ugraph = uTensorGraph(graph_def, output_nodes=output_nodes)
    transformer = DropoutTransformer()
    new_ugraph = transformer.transform(ugraph)
    for op in new_ugraph.ops_info.values():
        assert op.ugraph
        assert not op.is_dangling
    out_op = new_ugraph.ops_info[output_nodes[0]]
    assert set([str(op.name) for op in out_op.input_nodes]) == set(['x', 'bias'])
    # all dropout nodes should be gone
    graph_1 = tf.Graph()
    graph_2 = tf.Graph()
    with graph_1.as_default():
        tf.import_graph_def(ugraph.graph_def, name='')
    with graph_2.as_default():
        tf.import_graph_def(new_ugraph.graph_def, name='')
    with tf.Session(graph=graph_1) as sess:
        keep_prob = graph_1.get_tensor_by_name(keep_prob_name)
        dropout_output = graph_1.get_tensor_by_name(dropout_output_name)
        output = graph_1.get_tensor_by_name(output_nodes[0]+":0")
        # test the dropout ops are gone
        assert keep_prob.op.name not in new_ugraph.ops_info
        assert dropout_output.op.name not in new_ugraph.ops_info
        output_1 = output.eval({keep_prob:1.0})
    with tf.Session(graph=graph_2) as sess:
        output = graph_2.get_tensor_by_name(output_nodes[0]+":0")
        output_2 = output.eval()
    # expecting the same outputs with keep_prob == 1.0
    assert (output_1 == output_2).all()
