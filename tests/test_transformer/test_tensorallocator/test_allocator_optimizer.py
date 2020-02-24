from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.transformer import TensorLifeProbe


def test_create_resource_table(refgraph_tuple):
    (graph_def, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser(config={}).parse(graph_def, output_nodes=output_nodes)
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

def test_create_allocate_table(refgraph_tuple):
    (graph_def, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser(config={}).parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    table = transformer._create_resource_table(ugraph)
    allocate_table = dict()
    l = ugraph.topo_order[3]
    g = ugraph.ops_info[l]
    x = g.output_tensors[0]
    result = transformer._update_allocation_table(allocate_table, table, x, 0, 5)
    assert result['w/y:0']['offsetstart'] == 0
    assert result['w/y:0']['offsetend'] == 5



def test_query_offset_address(refgraph_tuple):
    (graph_def, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser(config={}).parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    table = transformer._create_resource_table(ugraph)
    allocate_table = dict()
    l = ugraph.topo_order[3]
    g = ugraph.ops_info[l]
    x = g.output_tensors[0]
    result = transformer._update_allocation_table(allocate_table, table, x, 0, 5)
    start, end = transformer._query_offset_fromallocate_table(allocate_table, 1, 5)
    assert start == 0
    assert end == 5

def test_query_timeline(refgraph_tuple):
    (graph_def, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser(config={}).parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    table = transformer._create_resource_table(ugraph)
    allocate_table = dict()
    l = ugraph.topo_order[3]
    g = ugraph.ops_info[l]
    x = g.output_tensors[0]
    result = transformer._update_allocation_table(allocate_table, table, x, 0, 5)
    start, end = transformer._query_time_fromallocate_table(allocate_table, 1, 5)
    assert start == 1
    assert end == 5

def test_query_result(refgraph_tuple):
    (graph_def, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser(config={}).parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    table = transformer._create_resource_table(ugraph)
    allocate_table = dict()
    l = ugraph.topo_order[0]
    g = ugraph.ops_info[l]
    x = g.output_tensors[0]
    result = transformer._update_allocation_table(allocate_table, table, x, 5, 10)
    l = ugraph.topo_order[1]
    g = ugraph.ops_info[l]
    y = g.output_tensors[0]
    #address and time overlap
    s = transformer._query_result(allocate_table, 3, 6, 1, 2)
    assert s
    s = transformer._query_result(allocate_table, 6, 8, 1, 2)
    assert s
    s = transformer._query_result(allocate_table, 9, 11, 1, 2)
    assert s
    #address overlap, but time doesn't
    s = transformer._query_result(allocate_table, 3, 6, 5, 6)
    assert not s
    #time overlap, but address doesn't
    s = transformer._query_result(allocate_table, 3, 4, 1, 2)
    assert s

def test_allocate_tensor(refgraph_tuple):
    (graph_def, output_nodes) = refgraph_tuple
    ugraph = GraphDefParser(config={}).parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    tensors = []
    table = transformer._create_resource_table(ugraph)
    allocate_table = dict()
    l = ugraph.topo_order[3]
    g = ugraph.ops_info[l]
    x = g.output_tensors[0]
    tensors.append(x)
    unit_size = 4
    buffer_size = 30000 #1k bytes
    result = transformer.allocate_tensor(tensors, 0, allocate_table, table, buffer_size, unit_size)

    assert result == True
    
def test_allocate_graph(refgraph_tuple):
  (graph_def, output_nodes) = refgraph_tuple
  ugraph = GraphDefParser(config={}).parse(graph_def, output_nodes=output_nodes)
  transformer = TensorLifeProbe()
  use_def_table = transformer._create_resource_table(ugraph)
  unit_size = 4
  buffer_size = 3000 #1k bytes
  allocate_table = dict()
  result = transformer.allocate_graph(ugraph, allocate_table, use_def_table, buffer_size, unit_size)
  assert result == True


def test_query_check(refgraph_tuple):
    (graph_def, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser(config={}).parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    table = transformer._create_resource_table(ugraph)
    allocate_table = dict()
    l = ugraph.topo_order[3]
    g = ugraph.ops_info[l]
    x = g.output_tensors[0]
    result = transformer._update_allocation_table(allocate_table, table, x, 0, 5)
    l = ugraph.topo_order[1]
    g = ugraph.ops_info[l]
    y = g.output_tensors[0]
    valid = transformer._check(result, table, y, 4, 10)
    assert valid == False

def test_memory_allocation(refgraph_tuple):
    (graph_def, output_nodes)= refgraph_tuple
    ugraph = GraphDefParser(config={}).parse(graph_def, output_nodes=output_nodes)
    transformer = TensorLifeProbe()
    ugraph = transformer.transform(ugraph)
