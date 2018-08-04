import tensorflow as tf
from utensor_cgen.ir import uTensorGraph, OperationInfo
from copy import deepcopy

def get_ops_io_info(op_type):
#please refer to OperatorFactory() in operators.py
  ops_io_table = dict()
  ops_io_table["Add"] =                 [[0, 0], [0]]
  ops_io_table["ArgMax"] =              [[0, 1], [0]]
  ops_io_table["Dequantize"] =          [[0, 1, 2], [0]]
  ops_io_table["Max"] =                 [[0, 1], [0]]
  ops_io_table["QuantizedMaxPool"] =    [[0, 1, 2], [0, 1, 2]]
  ops_io_table["Min"] =                 [[0, 1], [0]]
  ops_io_table["QuantizeV2"] =          [[0, 1, 2], [0, 1, 2]]
  ops_io_table["QuantizedMatMul"] =     [[0, 1, 2, 3, 4, 5], [0, 1, 2]]
  ops_io_table["QuantizedRelu"] =       [[0, 1, 2], [0, 1, 2]]
  ops_io_table["QuantizedAdd"] =        [[0, 0, 1, 2, 4, 5], [0, 1, 2]]
  ops_io_table["RequantizationRange"] = [[0, 1, 2], [0, 1]]
  ops_io_table["Requantize"] =          [[0, 1, 2, 3, 4], [0, 1, 2]]
  ops_io_table["Reshape"] =             [[0, 1], [0]]
  ops_io_table["QuantizedReshape"] =    [[0, 1, 2, 3], [0, 1, 2]]
  ops_io_table["QuantizedConv2D"] =     [[0, 1, 2, 3, 4, 5], [0, 1, 2]]
  ops_io_table["Const"] =               [None, [0]]
  ops_io_table["Placeholder"] =         [None, [0]]
  ops_io_table["Inline"] =              [None, [0]]

  return ops_io_table[op_type]


def print_topo_order(graph):
  for it_node in graph.topo_order:
    #print(it_node)
    print(it_node, " : ", graph.ops_info[it_node].op_type, "| ", [t.name for t in graph.ops_info[it_node].input_tensors], ":", [t.name for t in graph.ops_info[it_node].output_tensors])
  [exp_inputs, exp_outputs] = subgraph_trace_exposed_edges(graph)
  print("exposed inputs", exp_inputs)
  print("exposed outputs", exp_outputs)

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

def subgraph_trace_exposed_edges(graph, start_index=0, end_index=None):
  if end_index == None:
    end_index = len(graph.topo_order)

  sub_topo = graph.topo_order[start_index:end_index]

  subgraph_tensors_in = set()
  subgraph_tensors_out = set()

  for node in sub_topo:
    subgraph_tensors_in.update(get_input_tensors(graph, node))
    subgraph_tensors_out.update(get_output_tensors(graph, node))

  # print("subgraph_trace_exposed_edges: ", start_index, ", ", end_index)
  # print("original topo: ", graph.topo_order)
  # print("sub_topo: ", sub_topo)
  # print("subgraph_tensors_in: ", subgraph_tensors_in)
  # print("subgraph_tensors_out: ", subgraph_tensors_out)

  input_edges = subgraph_tensors_in.difference(subgraph_tensors_out)
  output_edges = subgraph_tensors_out.difference(subgraph_tensors_in)

  input_edges_list = list()
  output_edges_list = list()

  #ensure this follows topological order
  for node in sub_topo:
    for t_name in get_input_tensors(graph, node):
      if t_name in input_edges:
        input_edges_list.append(t_name)
    for t_name in get_output_tensors(graph, node):
      if t_name in output_edges:
        output_edges_list.append(t_name)

  return [input_edges_list, output_edges_list]

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
  print("matched indexi: ", start_index, ", ", end_index)
  [input_tensors, output_tensors] = subgraph_trace_exposed_edges(graph, start_index, end_index+1)
  [dropin_input_tensors, dropin_output_tensors] = subgraph_trace_exposed_edges(dropin_subgraph)
  assert (len(input_tensors) == len(dropin_input_tensors) and len(output_tensors) == len(dropin_output_tensors))
  #TODO:
  graph_backup = deepcopy(graph)
  for node in graph.topo_order[start_index:(end_index+1)]:
    remove_node(graph, node)

  graph.topo_order[start_index:start_index] = dropin_subgraph.topo_order
  #graph.topo_order[start_index:start_index] = [dropin_subgraph.topo_order[0]] #single node support
  #support fusing into a single node for now
  #TODO:
  #optimization pass: generate input/output attribute for each op
  #optimization pass: implement topological signature
  #ugraph class: holds metadata: applied passes, states
  #pass manager/config

  # input_tensor_infos = [tensorInfo_from_name(graph_backup, t_name) for t_name in input_tensors]
  # output_tensor_infos = [tensorInfo_from_name(graph_backup, t_name) for t_name in output_tensors]
  
  for dropin_node_name in dropin_subgraph.topo_order:
    new_op_type = dropin_subgraph.ops_info[dropin_node_name].op_type
    ops_io_info = get_ops_io_info(new_op_type)
    new_input_tensor_infos = list()
    new_output_tensor_infos = list()

    if ops_io_info[0] != None:
      new_input_tensor_infos = dropin_subgraph.ops_info[dropin_node_name].input_tensors #direct copy from the drop-in graph
      for i, tensor_info in enumerate(new_input_tensor_infos):
        if tensor_info.name in dropin_input_tensors: #contains an exposed edge
          drop_in_tensor_index = dropin_input_tensors.index(tensor_info.name) #exposed edge index matching with the subject graph
          cooresponding_subject_graph_tensor_name = input_tensors[drop_in_tensor_index]
          new_input_tensor_infos[i] = tensorInfo_from_name(graph_backup, cooresponding_subject_graph_tensor_name)


    if ops_io_info[1] != None:
      new_output_tensor_infos = dropin_subgraph.ops_info[dropin_node_name].output_tensors
      for i, tensor_info in enumerate(new_output_tensor_infos):
        if tensor_info.name in dropin_output_tensors:
          drop_in_tensor_index = dropin_output_tensors.index(tensor_info.name)
          cooresponding_subject_graph_tensor_name = output_tensors[drop_in_tensor_index]
          new_output_tensor_infos[i] = tensorInfo_from_name(graph_backup, cooresponding_subject_graph_tensor_name)

    new_op_info = OperationInfo(name=dropin_subgraph.ops_info[dropin_node_name].name,
                            input_tensors=new_input_tensor_infos,
                            output_tensors=new_output_tensor_infos,
                            op_type=dropin_subgraph.ops_info[dropin_node_name].op_type,
                            backend=dropin_subgraph.ops_info[dropin_node_name].backend,
                            op_attr=deepcopy(dropin_subgraph.ops_info[dropin_node_name].op_attr))
    graph.ops_info[new_op_info.name] = new_op_info

    ##TODO: trigger topo rebuild here

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
      if(matcher_index >= matcher_seq_len - 1):
        #matched
        return [start_index, i] #start index, end index
      matcher_index += 1
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
  print("======usubgraph content")
  print_topo_order(usubgraph)
  print("======ureplacement content")
  print_topo_order(ureplacement)
  print("==================")
  print("==================")
  assert subgraph_replace(ugraph, usubgraph, ureplacement), "pattern not found"
  assert compare_topological_orders(ugraph, uexpected), "ugraph does not equal to uexpected"
  print("==================")
  print("==================")
  print("======result content")
  print_topo_order(ugraph)
  print("======uexpected content")
  print_topo_order(uexpected)
  assert False