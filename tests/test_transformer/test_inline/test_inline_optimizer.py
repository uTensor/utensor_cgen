from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.transformer import InlineTransformer


def test_inline_optimizer(inlinegraph_tuple):
    (graph_def, inline_ans, output_nodes)=  inlinegraph_tuple
    ugraph = GraphDefParser.parse(graph_def, output_nodes)
    transformer = InlineTransformer()
    ugraph = transformer.transform(ugraph)
    for node_name in ugraph.topo_order:
        if node_name in inline_ans:
            op_type = ugraph.ops_map[node_name].op_type
            assert op_type == 'Inline'
