import tensorflow as tf
from utensor_cgen.ir import uTensorGraph

def compare_topological_orders(graph0, graph1):
  if(len(graph0.topo_order) != len(graph1.topo_order)):
    return False

  for node0, node1 in zip(graph0.topo_order, graph1.topo_order):
    op_type0 = graph0.ops_info[node0].op_type
    op_type1 = graph1.ops_info[node1].op_type
    if(op_type0 != op_type1):
      return False
  return True

def test_fusion_support_functions(fusion_graph_tuple):
  (ugraph, usubgraph, ureplacement, uexpected) = fusion_graph_tuple
  #requires: nextNodes(node), prevNodes(node)
  #requires: node_replace(node)
  assert compare_topological_orders(ugraph, ugraph), "ugraph and ugraph should be equal"
  assert not compare_topological_orders(ugraph, usubgraph), "ugraph and ugraph_tuple are not equal"

def test_fusion_transformer(fusion_graph_tuple):
  (ugraph, usubgraph, ureplacement, uexpected) = fusion_graph_tuple
  #requires: nextNodes(node), prevNodes(node)
  #requires: node_replace(node)
  assert compare_topological_orders(ugraph, ugraph), "ugraph and ugraph should be equal"
  assert not compare_topological_orders(ugraph, usubgraph), "ugraph and ugraph_tuple are not equal"