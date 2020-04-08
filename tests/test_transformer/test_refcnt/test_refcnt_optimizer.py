from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.transformer import RefCntOptimizer


def test_refcnt_optimizer(refgraph_tuple):
    (graph_def, refcnt_ans, output_nodes) = refgraph_tuple
    ugraph = GraphDefParser(config={}).parse(graph_def, output_nodes=output_nodes)
    transformer = RefCntOptimizer()
    assert not transformer.prune_graph
    ugraph = transformer.transform(ugraph)
    for node_name in ugraph.topo_order:
        if node_name in refcnt_ans:
            op_info = ugraph.ops_info[node_name]
            refcnts = op_info.op_attr["%s__ref_counts" % transformer.KWARGS_NAMESCOPE]
            assert refcnts == refcnt_ans[node_name]
