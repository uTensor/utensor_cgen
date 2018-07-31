import tensorflow as tf
from utensor_cgen.ir import uTensorGraph, OperationInfo
from copy import deepcopy

def print_topo_order(graph):
  for it_node in graph.topo_order:
    #print(it_node)
    print(it_node, " : ", graph.ops_info[it_node].op_type)

def compare_topological_orders(graph0, graph1):
  if(len(graph0.topo_order) != len(graph1.topo_order)):
    print("======graph0 topo")
    print_topo_order(graph0)
    print("======graph1 topo")
    print_topo_order(graph1)
    return False

  for node0, node1 in zip(graph0.topo_order, graph1.topo_order):
    op_type0 = graph0.ops_info[node0].op_type
    op_type1 = graph1.ops_info[node1].op_type
    if(op_type0 != op_type1):
      print("======graph0 topo")
      print_topo_order(graph0)
      print("======graph1 topo")
      print_topo_order(graph1)
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

def get_input_tensors(graph, node_name):
  return [t.name for t in graph.ops_info[node_name].input_tensors]

def get_output_tensors(graph, node_name):
  return [t.name for t in graph.ops_info[node_name].output_tensors]

def subgraph_trace_exposed_edges(graph, start_index = 0, end_index=-2):
  sub_topo = graph.topo_order[start_index:(end_index+1)]
  subgraph_tensors_in = set()
  subgraph_tensors_out = set()

  for node in sub_topo:
    subgraph_tensors_in.update(get_input_tensors(graph, node))
    subgraph_tensors_out.update(get_output_tensors(graph, node))

  input_edges = subgraph_tensors_in.difference(subgraph_tensors_out)
  output_edges = subgraph_tensors_out.difference(subgraph_tensors_in)

  return [input_edges, output_edges]

# def edge_cleanup(graph, edge_name):
#   for node in graph.topo_order:
#     tensors_in = get_input_tensors(graph, node)
#     tensors_out = get_output_tensors(graph, node)
#     if(edge_name in tensors_in or edge_name in tensors_out):
#       return
#   # TODO: delete the edge here


def remove_node(graph, node_name):
  graph.topo_order.remove(node_name)
  #graph.ops_info.remove(node_name)
  del graph.ops_info[node_name]
  #trigger edge recount  ## probably don't need this

def tensorInfo_from_name(graph, edge_name):
  for it_node in graph.topo_order:
    for t in graph.ops_info[it_node].input_tensors:
      if t.name == edge_name:
        return t
    for t in graph.ops_info[it_node].output_tensors:
      if t.name == edge_name:
        return t

  return None

def subgraph_replace(graph, matcher_subgraph, dropin_subgraph):
  matched = subgraph_latch(graph, matcher_subgraph, dropin_subgraph)
  if(matched == None):
    return False
  [start_index, end_index] = matched
  [input_tensors, output_tensors] = subgraph_trace_exposed_edges(graph, start_index, end_index)
  [dropin_input_tensors, dropin_output_tensors] = subgraph_trace_exposed_edges(dropin_subgraph)
  assert (len(input_tensors) != len(dropin_input_tensors) or len(output_tensors) != len(dropin_output_tensors))
  #TODO:
  graph_backup = deepcopy(graph)
  for node in graph.topo_order[start_index:(end_index+1)]:
    remove_node(graph, node)

  #graph.topo_order[start_index:start_index] = dropin_subgraph.topo_order
  graph.topo_order[start_index:start_index] = [dropin_subgraph.topo_order[0]] #single node support
  #support fusing into a single node for now
  #TODO:
  #optimization pass: generate input/output attribute for each op
  #optimization pass: implement topological signature
  #ugraph class: holds metadata: applied passes, states
  #pass manager/config
  assert (len(dropin_subgraph.topo_order) != 1), "only fusing to 1 node is implemented"
  dropin_node_name = dropin_subgraph.topo_order[0]
  input_tensor_infos = [tensorInfo_from_name(graph_backup, t_name) for t_name in input_tensors]
  output_tensor_infos = [tensorInfo_from_name(graph_backup, t_name) for t_name in output_tensors]
  new_op_info = OperationInfo(name=dropin_subgraph.ops_info[dropin_node_name].name,
                            input_tensors=input_tensor_infos,
                            output_tensors=output_tensor_infos,
                            op_type=dropin_subgraph.ops_info[dropin_node_name].op_type,
                            backend=dropin_subgraph.ops_info[dropin_node_name].backend,
                            op_attr=deepcopy(dropin_subgraph.ops_info[dropin_node_name].op_attr))
  graph.ops_info[new_op_info.name] = new_op_info
  return True


def subgraph_latch(graph, matcher_subgraph, dropin_subgraph):
  topo = graph.topo_order
  match_topo = matcher_subgraph.topo_order
  matcher_index = 0
  matcher_seq_len = len(match_topo)
  start_index = 0 #keep a record of where the match starts
  for i, node in enumerate(topo):
    matcher_node = match_topo[matcher_index]
    #TODO: need to implement a connectivity test
    if(graph.ops_info[node].op_type == matcher_subgraph.ops_info[matcher_node].op_type):
      matcher_index += 1
      if(matcher_index >= matcher_seq_len):
        #matched
        return [start_index, i] #start index, end index
    else:
      matcher_index = 0
      start_index = i
  return None

def test_fusion_support_functions(fusion_graph_tuple):
  (ugraph, usubgraph, ureplacement, uexpected) = fusion_graph_tuple
  assert compare_topological_orders(ugraph, ugraph), "ugraph and ugraph should be equal"
  assert not compare_topological_orders(ugraph, usubgraph), "ugraph and ugraph_tuple are not equal"
  assert is_connected(ugraph, "node_add0", "node_add1"), "node_add0 and node_add1 in ugraph should be connected"
  assert not is_connected(ugraph, "node_add0", "node_add2"), "node_add0 and node_add2 in ugraph shouldn't be connected"

def test_fusion_transformer(fusion_graph_tuple):
  (ugraph, usubgraph, ureplacement, uexpected) = fusion_graph_tuple
  print("======ugraph content")
  print_topo_order(ugraph)
  assert subgraph_replace(ugraph, usubgraph, ureplacement), "pattern not found"
  print("======usubgraph content")
  print_topo_order(usubgraph)
  print("======ureplacement content")
  print_topo_order(ureplacement)
  assert compare_topological_orders(ugraph, uexpected), "ugraph does not equal to uexpected"