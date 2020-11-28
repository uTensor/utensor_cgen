import tensorflow.compat.v1 as tf

# FIXME: remove uTensorOpEqualityDelegate import after we have generic op_eq_deleate
from utensor_cgen.backend.utensor.code_generator.rearch._operators import \
    uTensorOpEqualityDelegate
from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.matcher import uTensorGraphMatcher
from utensor_cgen.utils import prune_graph, topologic_order_graph


def test_replace_fc_with_add(subj_graph_1, patrn_fc_1):
    def callback(match):
        graph = tf.Graph()
        with graph.as_default():
            a = tf.placeholder(dtype=tf.float32, name='a')
            b = tf.placeholder(dtype=tf.float32, name='b')
            out = tf.add(a, b, name='fused_node')
        ugraph = GraphDefParser(config={}).parse(graph.as_graph_def(), output_nodes=[out.op.name])
        ugraph.ops_info['fused_node'].replace_with_null_input_tensor(0)
        ugraph.ops_info['fused_node'].replace_with_null_input_tensor(1)
        topologic_order_graph(ugraph)
        ugraph = prune_graph(ugraph)
        patrn_ugraph = match.pattern_ugraph
        
        input_map = {
            patrn_ugraph.ops_info['a_prime'].input_tensors[0]: ugraph.ops_info['fused_node'].input_tensors[0],
            patrn_ugraph.ops_info['a_prime'].input_tensors[1]: ugraph.ops_info['fused_node'].input_tensors[1]
        }
        output_map = {
            patrn_ugraph.ops_info['r_prime'].output_tensors[0]: ugraph.ops_info['fused_node'].output_tensors[0]
        }
        return ugraph, input_map, output_map
    matcher = uTensorGraphMatcher(patrn_fc_1, op_equality_delegate=uTensorOpEqualityDelegate)
    matches = matcher.match(subj_graph_1)
    assert matches, 'no match found'
    match = matches[0]
    new_ugraph = match.replace_with(callback)
    test_pass = True
    missed_op_names = []
    for op_name in match.subj2patrn_op_map:
        if op_name in new_ugraph.ops_info:
            test_pass = False
            missed_op_names.append(op_name)
    assert test_pass, \
        'these ops should not be found in the new ugrah: {}'.format(missed_op_names)
