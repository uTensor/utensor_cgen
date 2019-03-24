from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.transformer import TensorLifeProbe


def test_create_resource_table(refgraph_tuple):
    (graph_def, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser.parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    table = transformer._create_resource_table(ugraph)
    resource_ans = {
        'x:0' : [0, 4],
        'y:0' : [1, 2],
        'z:0' : [2, 5],
        'w/y:0': [3, 4],
        'w:0' : [4, 5],
        'k:0' : [5, 5]
    }
    for t in table:
        assert table[t]['start'] == resource_ans[t][0]
        assert table[t]['end'] == resource_ans[t][1]
'''
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
    #print(result)

def test_query_offset_address(refgraph_tuple):
    (graph_def, refcnt_ans, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser.parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    table = transformer._create_resource_table(ugraph)
    allocate_table = dict()
    l = ugraph.topo_order[3]
    g = ugraph.ops_info[l]
    x = g.output_tensors[0]
    result = transformer._create_allocate_table(allocate_table, table, x, 0, 5)
    start, end = transformer._query_offset_fromallocate_table(allocate_table, 1, 5)
    print(start)
    print(end)

def test_query_timeline(refgraph_tuple):
    (graph_def, refcnt_ans, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser.parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    table = transformer._create_resource_table(ugraph)
    allocate_table = dict()
    l = ugraph.topo_order[3]
    g = ugraph.ops_info[l]
    x = g.output_tensors[0]
    result = transformer._create_allocate_table(allocate_table, table, x, 0, 5)
    start, end = transformer._query_time_fromallocate_table(allocate_table, 1, 5)
    print(start)
    print(end)

def test_query_result(refgraph_tuple):
    (graph_def, refcnt_ans, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser.parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    table = transformer._create_resource_table(ugraph)
    allocate_table = dict()
    l = ugraph.topo_order[3]
    g = ugraph.ops_info[l]
    x = g.output_tensors[0]
    print("tensor")
    print(x)
    result = transformer._create_allocate_table(allocate_table, table, x, 0, 5)
    start, end = transformer._query_time_fromallocate_table(allocate_table, 1, 5)
    s = transformer._query_result(allocate_table, x, 0, 5, 2, 4)
    print(s[0])

def test_query_check(refgraph_tuple):
    (graph_def, refcnt_ans, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser.parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    table = transformer._create_resource_table(ugraph)
    allocate_table = dict()
    l = ugraph.topo_order[3]
    g = ugraph.ops_info[l]
    x = g.output_tensors[0]
    result = transformer._create_allocate_table(allocate_table, table, x, 0, 5)
    valid = transformer._check(ugraph, result, table, x, 6, 10)
    print(valid)

def test_memory_allocation(refgraph_tuple):
    (graph_def, refcnt_ans, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser.parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    valid = transformer.transform(ugraph)
    #print(valid)
'''
