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

def get_input_nodes(graph, node):
  tensors_in = set([t.name for t in graph.ops_info[node].input_tensors])
  node_list = set()
  for it_node in graph.topo_order:
    if(it_node == node):
      continue
    it_tensors_out = [t.name for t in graph.ops_info[it_node].output_tensors]
    if not tensors_in.isdisjoint(it_tensors_out):
      node_list.add(it_node)

  return node_list

def get_output_nodes(graph, node):
  tensors_out = set([t.name for t in graph.ops_info[node].output_tensors])
  node_list = set()
  for it_node in graph.topo_order:
    if(it_node == node):
      continue
    it_tensors_in = [t.name for t in graph.ops_info[it_node].input_tensors]
    if not tensors_out.isdisjoint(it_tensors_in):
      node_list.add(it_node)

  return node_list

def is_connected(graph, node0, node1):
  input_nodes = get_input_nodes(graph, node0)
  output_nodes = get_output_nodes(graph, node0)
  node_list = input_nodes.union(output_nodes)

  return node1 in node_list

def test_fusion_support_functions(fusion_graph_tuple):
  (ugraph, usubgraph, ureplacement, uexpected) = fusion_graph_tuple
  assert compare_topological_orders(ugraph, ugraph), "ugraph and ugraph should be equal"
  assert not compare_topological_orders(ugraph, usubgraph), "ugraph and ugraph_tuple are not equal"
  assert is_connected(ugraph, "node_add0", "node_add1"), "node_add0 and node_add1 in ugraph should be connected"
  assert not is_connected(ugraph, "node_add0", "node_add2"), "node_add0 and node_add2 in ugraph shouldn't be connected"

def test_fusion_transformer(fusion_graph_tuple):
  (ugraph, usubgraph, ureplacement, uexpected) = fusion_graph_tuple
  #requires: nextNodes(node), prevNodes(node)
  #requires: node_replace(node)
  assert compare_topological_orders(ugraph, ugraph), "ugraph and ugraph should be equal"
  assert not compare_topological_orders(ugraph, usubgraph), "ugraph and ugraph_tuple are not equal"