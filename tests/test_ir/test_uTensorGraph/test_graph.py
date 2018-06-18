from copy import deepcopy

from utensor_cgen.ir import uTensorGraph

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