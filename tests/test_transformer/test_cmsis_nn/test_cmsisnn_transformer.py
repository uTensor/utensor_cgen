from utensor_cgen.transformer import CMSIS_NN_Transformer
from utensor_cgen.ir import uTensorGraph

def graph_check(graph):
  for op_name, op_info in graph.ops_info.items():
    for input_tensor_info in op_info.input_tensors:
      assert input_tensor_info.op_name in graph.ops_info, "In %r: input tensor %r points to non-existing op %r" % (op_name, input_tensor_info.name, input_tensor_info.op_name)
      assert input_tensor_info.op_name in graph.topo_order

  assert len(graph.ops_info) == len(graph.topo_order)

def test_cmsisnn_trnasformer(fusion_graph_tuple):
    (ugraph, ugraph1) = fusion_graph_tuple
    transformer = CMSIS_NN_Transformer()
    test_graph = transformer.transform(ugraph1)
    graph_check(test_graph)
    print(test_graph.topo_order)
