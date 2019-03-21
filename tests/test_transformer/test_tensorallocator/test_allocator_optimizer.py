from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.transformer import TensorLifeProbe


def test_create_resource_table(refgraph_tuple):
    (graph_def, refcnt_ans, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser.parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    table = transformer._create_resource_table(ugraph)
    print(table)


def test_create_allocate_table(refgraph_tuple):
    (graph_def, refcnt_ans, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser.parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    table = transformer._create_resource_table(ugraph)
    allocate_table = dict()
    l = ugraph.topo_order[3]
    g = ugraph.ops_info[l]
    x = g.output_tensors[0]
    result = transformer._create_allocate_table(allocate_table, table, x, 0, 5)
    print(result)
