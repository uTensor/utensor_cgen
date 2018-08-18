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

#depth = 1 only checks the current node
def node_forward_isomorph(subject_node_name, matcher_node_name, subject_graph, matcher_graph, depth=None):
  matcher_to_subject_nodes = dict()
  matcher_to_subject_edges = dict()

  if depth == None:
    depth = len(matcher_graph.ops_info)-1

  subject_node = subject_graph.ops_info[subject_node_name]
  matcher_node = matcher_graph.ops_info[matcher_node_name]

  if subject_node.op_type != matcher_node.op_type:
    return False

  #check next level
  ## the case when there's no next level or when depth is reached
  if len(get_output_nodes(matcher_graph, matcher_node_name)) <= 0 or depth <= 0:
    #translate and return the nodes and edges
    matcher_node_output_tensors = get_output_tensors(matcher_graph, matcher_node_name)
    subject_node_output_tensors = get_output_tensors(subject_graph, subject_node_name)
    for i, matcher_node_output_tensor in enumerate(matcher_node_output_tensors):
      matcher_to_subject_edges[matcher_node_output_tensor] = subject_node_output_tensors[i]
    matcher_to_subject_nodes[matcher_node_name] = subject_node_name
    return [matcher_to_subject_nodes, matcher_to_subject_edges]

  [_, output_groups] = get_ops_io_info(matcher_node.op_type)

  matcher_output_node_names = set()
  subject_output_node_names = set()

  #through the number of groups
  for group in list(range(0,max(output_groups)+1)):

    #for the same group
    for output_index, group_id in enumerate(output_groups):
      if group_id != group:
        continue
      
      for output_inner_index, group_inner_id in enumerate(output_groups):
        if group_inner_id != group:
          continue
        
        matcher_tensor_name = matcher_node.output_tensors[output_inner_index].name
        subject_tensor_name = subject_node.output_tensors[output_inner_index].name
        
        if not matcher_tensor_name in matcher_to_subject_edges:
          #adding all connected node of the same group to a set
          [_, tmp] = get_tensor_node_names(matcher_graph, matcher_tensor_name) 
          matcher_output_node_names.update(tmp)

          [_, tmp] = get_tensor_node_names(subject_graph, subject_tensor_name) 
          subject_output_node_names.update(tmp)
      #all unevaluated outputs are added to the lists at the point


      #matching a output group
      result = False
      for matcher_output_node in matcher_output_node_names:
        for subject_output_node in subject_output_node_names:
          result = node_forward_isomorph(subject_output_node, matcher_output_node, subject_graph, matcher_graph, depth-1)
          #early termination
          if result != False:
            #combining the lists
            matcher_to_subject_nodes.update(result[0]) #updating a dicitionary
            matcher_to_subject_edges.update(result[1])
            break
        if result != False:
          break
      #if no viable combo is found
      if result == False:
        return False
  #all group outputs matched
  return [matcher_to_subject_nodes, matcher_to_subject_edges]

#   return [matcher_to_subject_nodes, matcher_to_subject_edges]

def isomorphic_match(subject_graph, matcher_graph):
  matcher_to_subject_nodes = dict()
  matcher_to_subject_edges = dict()
  for matcher_node_name in matcher_graph.topo_order:
    if not matcher_node_name in matcher_to_subject_nodes:
      for subject_node_name in subject_graph.topo_order:
        subject_node = subject_graph.ops_info[subject_node_name]
        matcher_node = matcher_graph.ops_info[matcher_node_name]
        if subject_node.op_type == matcher_node.op_type:
          result = node_forward_isomorph(subject_node_name, matcher_node_name, subject_graph, matcher_graph)
          if result != False:
            matcher_to_subject_nodes.update(result[0]) #updating a dicitionary
            matcher_to_subject_edges.update(result[1])
  for matcher_node_name in matcher_graph.topo_order:
    if not matcher_node_name in matcher_to_subject_nodes:
      return False
  
  return [matcher_to_subject_nodes, matcher_to_subject_edges]


def subgraph_latch(graph, matcher_subgraph):
  topo = graph.topo_order
  match_topo = matcher_subgraph.topo_order
  matcher_seq_len = len(match_topo)

  for i in range(0,len(topo) - matcher_seq_len):
    search_topo = topo[i:(i+matcher_seq_len)]
    for j, test_node in enumerate(search_topo):
      matcher_node = match_topo[j]
      if(graph.ops_info[test_node].op_type != matcher_subgraph.ops_info[matcher_node].op_type):
        break
      if(j >= matcher_seq_len-1): #matched
        return [i, i + j]

  return None

def test_fusion_support_functions(fusion_graph_tuple):
  (ugraph, usubgraph, ureplacement, uexpected) = fusion_graph_tuple
  assert True, "assert True"
  assert compare_topological_orders(ugraph, ugraph), "ugraph and ugraph should be equal"
  assert not compare_topological_orders(ugraph, usubgraph), "ugraph and ugraph_tuple are not equal"
  assert is_connected(ugraph, "node_add0", "node_add1"), "node_add0 and node_add1 in ugraph should be connected"
  assert not is_connected(ugraph, "node_add0", "node_add2"), "node_add0 and node_add2 in ugraph shouldn't be connected"
  assert subgraph_latch(ugraph, usubgraph), "subgraph_latch failed"

def test_fusion_transformer(fusion_graph_tuple):
  (ugraph, usubgraph, ureplacement, uexpected) = fusion_graph_tuple
  print("======ugraph content")
  print_topo_order(ugraph)
  print("======usubgraph content")
  print_topo_order(usubgraph)
  import pdb; pdb.set_trace()
  assert isomorphic_match(ugraph, usubgraph), "pattern not found"
  #assert compare_topological_orders(ugraph, uexpected), "ugraph does not equal to uexpected"

def main():
  test_param = fusion_graph_tuple()
  print("testing support functions:")
  test_fusion_support_functions(test_param)
  print("testing support fusion_transformer:")
  test_fusion_transformer(test_param)

if __name__ == "__main__":
  main()