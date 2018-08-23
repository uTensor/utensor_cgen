import tensorflow as tf
import numpy as np
from utensor_cgen.ir import uTensorGraph

def fusion_graph_tuple():
  graph = tf.Graph()
  with graph.as_default():
    input0 = tf.placeholder(dtype=tf.float32,
                                   name='input0')
    input1 = tf.placeholder(dtype=tf.float32,
                                   name='input1')
    input2 = tf.placeholder(dtype=tf.float32,
                                   name='input2')
    node_add0 = tf.add(input0, input1, name="node_add0")
    node_add1 = tf.add(node_add0, input1, name="node_add1")
    node_add2 = tf.add(node_add1, input2, name="node_add2")

  ugraph = uTensorGraph(graph.as_graph_def(), [node_add2.name])
  #graph_tuple = (graph, (input0, input1), (node_add2))
#######
  subgraph = tf.Graph()
  with subgraph.as_default():
    subgraph_input0 = tf.placeholder(dtype=tf.float32,
                                   name='subgraph_input0')
    subgraph_input1 = tf.placeholder(dtype=tf.float32,
                                   name='subgraph_input1')
    subgraph_node_add0 = tf.add(subgraph_input0, subgraph_input1, name="subgraph_node_add0")
    subgraph_node_add1 = tf.add(subgraph_node_add0, subgraph_input1, name="subgraph_node_add1")
  
  usubgraph = uTensorGraph(subgraph.as_graph_def(), [subgraph_node_add1.name])
  #subgraph_tuple = (subgraph, (subgraph_input0, subgraph_input1), (subgraph_node_add1))
#######
  replacement_graph = tf.Graph()
  with replacement_graph.as_default():
    replacement_input0 = tf.placeholder(dtype=tf.float32,
                                   name='replacement_input0')
    replacement_input1 = tf.placeholder(dtype=tf.float32,
                                   name='replacement_input1')
    replacement_node_add0 = tf.add(replacement_input0, replacement_input1, name="replacement_node_add0")
  ureplacement = uTensorGraph(replacement_graph.as_graph_def(), [replacement_node_add0.name])
  #replacement_tuple = (replacement_graph, (replacement_input0, replacement_input1), (replacement_node_add0))
#######
  expected_graph = tf.Graph()
  with expected_graph.as_default():
    expected_input0 = tf.placeholder(dtype=tf.float32,
                                   name='expected_input0')
    expected_input1 = tf.placeholder(dtype=tf.float32,
                                   name='expected_input1')
    expected_input2 = tf.placeholder(dtype=tf.float32,
                                   name='expected_input2')
    expected_node_add0 = tf.add(expected_input0, expected_input1, name="expected_node_add0")
    expected_node_add1 = tf.add(expected_node_add0, expected_input2, name="expected_node_add1")
  uexpected = uTensorGraph(expected_graph.as_graph_def(), [expected_node_add1.name])
  #expected_tuple = (expected_graph, (expected_input0, expected_input1), (expected_node_add0))

  return (ugraph, usubgraph, ureplacement, uexpected)

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

def get_input_nodes(graph, node_name):
  tensors_in = set([t.name for t in graph.ops_info[node_name].input_tensors])
  node_list = set()
  for it_node in graph.topo_order:
    if(it_node == node_name):
      continue
    it_tensors_out = [t.name for t in graph.ops_info[it_node].output_tensors]
    if not tensors_in.isdisjoint(it_tensors_out):
      node_list.add(it_node)

  return node_list

def get_output_nodes(graph, node_name):
  tensors_out = set([t.name for t in graph.ops_info[node_name].output_tensors])
  node_list = set()
  for it_node in graph.topo_order:
    if(it_node == node_name):
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

def subgraph_trace_internal_edges(graph, start_index=0, end_index=None):
  if end_index == None:
    end_index = len(graph.topo_order)

  sub_topo = graph.topo_order[start_index:end_index]

  subgraph_tensors_in = set()
  subgraph_tensors_out = set()

  for node in sub_topo:
    subgraph_tensors_in.update(get_input_tensors(graph, node))
    subgraph_tensors_out.update(get_output_tensors(graph, node))
    
  internal_edges = subgraph_tensors_in.intersection(subgraph_tensors_out)

  internal_edges_list = list()

  #ensure this follows topological order
  for node in sub_topo:
    for t_name in get_input_tensors(graph, node):
      if t_name in internal_edges and not t_name in internal_edges_list:
        internal_edges_list.append(t_name)
    for t_name in get_output_tensors(graph, node):
      if t_name in internal_edges and not t_name in internal_edges_list:
        internal_edges_list.append(t_name)

  return internal_edges_list

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

#find the origin and destinations of a tensor
def get_tensor_node_names(graph, t_name):
  start_nodes = list()
  end_nodes = list()

  for it_node in graph.topo_order:
    for t in graph.ops_info[it_node].input_tensors:
      if t.name == t_name:
        end_nodes.append(it_node)
    for t in graph.ops_info[it_node].output_tensors:
      if t.name == t_name:
        start_nodes.append(it_node)

  return [start_nodes, end_nodes]

#named sequence
def forward_path_tracer(graph, start_node_name, end_node_name, depth=-1):
  # start_node = graph.ops_info[start_node_name]
  # end_node = graph.ops_info[end_node_name]
  output_node_names = set()
  path_list = list()

  if start_node_name == end_node_name:
    tmp = list()
    tmp.append(start_node_name)
    path_list.append(tmp)
    return path_list
  
  if depth == -1:
    depth = len(graph.topo_order) - 1

  output_node_names = get_output_nodes(graph, start_node_name)
  if len(output_node_names) == 0 or depth <= 0:
    return None
  
  for output_node_name in output_node_names:
    forward_path_list = forward_path_tracer(graph, output_node_name, end_node_name, depth-1)
    if forward_path_list == None:
      continue
    for forward_path in forward_path_list:
      forward_path.insert(0, start_node_name) #list
      path_list.append(forward_path) #list of list

  if len(path_list) == 0:
    return None

  return path_list

#recursive helper
#returns False when mismatch
#returns node-name-relation if a match exist
def isomorphic_associativity_helper(subject_node_name, matcher_node_name, subject_graph, matcher_graph, depth=0):
  #compare the current nodes
  subject_node = subject_graph.ops_info[subject_node_name]
  matcher_node = matcher_graph.ops_info[matcher_node_name]
  
  if subject_node.op_type != matcher_node.op_type:
    return False

  #create and concate the named-relations
  matcher_to_subject_nodes = dict()
  matcher_to_subject_edges = dict()
  matcher_to_subject_nodes[matcher_node_name] = subject_node_name

  matcher_node = matcher_graph.ops_info[matcher_node_name]
  [input_groups, _] = get_ops_io_info(matcher_node.op_type)
  #dealing with end of terminations
  if depth <= 0 or len(matcher_node.output_tensors) == 0 or input_groups == None:
    return [matcher_to_subject_nodes, matcher_to_subject_edges]

  used_input_id = set()

  for group in list(range(0,max(input_groups)+1)):
    #this loop refer to the matcher
    for input_id, group_id in enumerate(input_groups):
      if group_id != group:
        continue
      #currently iterating over the same group

      #sweeping all inputs with the current group_id
      #mark used when successful
      for sweeping_input_id, sweeping_group_id in enumerate(input_groups):
        if sweeping_group_id != group_id or sweeping_input_id in used_input_id:
          continue
        #sweep the subject input nodes
        sweeping_subject_input_tensor_names = get_input_tensors(subject_graph, subject_node_name)
        sweeping_subject_input_tensor_name = sweeping_subject_input_tensor_names[sweeping_input_id]
        [sweeping_subject_input_node_name, _] = get_tensor_node_names(subject_graph, sweeping_subject_input_tensor_name)

        matcher_input_tensor_names = get_input_tensors(matcher_graph, matcher_node_name)
        matcher_input_tensor_name = matcher_input_tensor_names[input_id]
        [matcher_input_node_name, _] = get_tensor_node_names(matcher_graph, matcher_input_tensor_name)

        # if subject_node.name == "node_add0" and matcher_node.name == "subgraph_node_add0":
        #   import pdb; pdb.set_trace()
        #   print("===================")

        probe = isomorphic_associativity_helper(sweeping_subject_input_node_name[0], matcher_input_node_name[0], subject_graph, matcher_graph, depth-1)
        if probe == False:
          continue
        #a match is found here:
        used_input_id.add(sweeping_input_id)
        matcher_to_subject_nodes.update(probe[0])
        matcher_to_subject_edges[matcher_input_tensor_name] = sweeping_subject_input_tensor_name
        matcher_to_subject_edges.update(probe[1])
  print("subject node: ", subject_node.name)
  print("matcher node: ", matcher_node.name)
  print("used_input_id vs input group: ", used_input_id, input_groups)
  # import pdb; pdb.set_trace()

  if len(used_input_id) == len(input_groups):
    return [matcher_to_subject_nodes, matcher_to_subject_edges]
  
  return False

def isomorphic_match(subject_graph, matcher_graph):
  matcher_to_subject_nodes = dict()
  matcher_to_subject_edges = dict()
  matcher_output_node_names = set()

  #identify matcher output nodes
  [_, matcher_output_edges] = subgraph_trace_exposed_edges(matcher_graph)
  
  for matcher_output_edge in matcher_output_edges: 
    [tensor_origin, _] = get_tensor_node_names(matcher_graph, matcher_output_edge)
    matcher_output_node_names.add(tensor_origin[0])
  max_search_depth = len(matcher_graph.topo_order)
  partial_matcher_to_subject_nodes = None
  partial_matcher_to_subject_edges = None
  #support only signle matcher output node for now
  for subject_node_name in subject_graph.topo_order:
    probe = isomorphic_associativity_helper(subject_node_name, next(iter(matcher_output_node_names)), subject_graph, matcher_graph, max_search_depth)
    if probe == False:
      continue
    [partial_matcher_to_subject_nodes, partial_matcher_to_subject_edges] = probe

  if partial_matcher_to_subject_nodes == None and partial_matcher_to_subject_edges == None:
    return False

  matcher_to_subject_nodes.update(partial_matcher_to_subject_nodes)
  matcher_to_subject_edges.update(partial_matcher_to_subject_edges)

  return [matcher_to_subject_nodes, matcher_to_subject_edges]

def test_fusion_support_functions(fusion_graph_tuple):
  (ugraph, usubgraph, ureplacement, uexpected) = fusion_graph_tuple
  assert True, "assert True"
  assert compare_topological_orders(ugraph, ugraph), "ugraph and ugraph should be equal"
  assert not compare_topological_orders(ugraph, usubgraph), "ugraph and ugraph_tuple are not equal"
  assert is_connected(ugraph, "node_add0", "node_add1"), "node_add0 and node_add1 in ugraph should be connected"
  assert not is_connected(ugraph, "node_add0", "node_add2"), "node_add0 and node_add2 in ugraph shouldn't be connected"
  # assert subgraph_latch(ugraph, usubgraph), "subgraph_latch failed"

def test_fusion_transformer(fusion_graph_tuple):
  (ugraph, usubgraph, ureplacement, uexpected) = fusion_graph_tuple
  print("======ugraph content")
  print_topo_order(ugraph)
  print("======usubgraph content")
  print_topo_order(usubgraph)
  result = isomorphic_match(ugraph, usubgraph)
  assert result, "pattern not found"
  print("=====================")
  print(result)
  #assert compare_topological_orders(ugraph, uexpected), "ugraph does not equal to uexpected"
  import pdb; pdb.set_trace()

def main():
  test_param = fusion_graph_tuple()
  print("testing support functions:")
  test_fusion_support_functions(test_param)
  print("testing support fusion_transformer:")
  test_fusion_transformer(test_param)

if __name__ == "__main__":
  main()