import numpy as np
from pytest import fixture

import tensorflow as tf
from utensor_cgen.ir import TensorInfo, OperationInfo, uTensorGraph
from utensor_cgen.utils import prune_graph, topologic_order_graph

@fixture(name='sample_ugraph')
def simple_tflm_graph():
    ugraph = uTensorGraph()

    weight_op = OperationInfo(
        name = "weight_const",
        op_type = "Const",
        backend = "tensorflow",
        ugraph = ugraph,
        op_attr = dict(),
        input_tensors = [],
        output_tensors = []
    )
    weight_op.op_attr["value"] = np.array([1,2,3,4], dtype=np.int8)
    weight_op.op_attr["shape"] = [4,1]

    weight = TensorInfo(
        name = "weight",
        op_name = "weight_const",
        dtype = np.dtype("int8"),
        shape = weight_op.op_attr["shape"],
        ugraph = ugraph
    )
    weight_op.output_tensors = [weight]

    mock_input_op = OperationInfo(
        name = "mock_input_const",
        op_type = "Const",
        backend = "tensorflow",
        ugraph = ugraph,
        op_attr = dict(),
        input_tensors = [],
        output_tensors = []
    )
    mock_input_op.op_attr["value"] = np.array([[1],[2],[3],[4]], dtype=np.int8)
    mock_input_op.op_attr["shape"] = [1,4]

    input1 = TensorInfo(
        name = "input",
        op_name = "mock_input_const",
        dtype = np.dtype("float"),
        shape = [1, 4],
        ugraph = ugraph
    )

    mock_input_op.output_tensors = [input1]

    bias_op = OperationInfo(
        name = "bias_const",
        op_type = "Const",
        backend = "tensorflow",
        ugraph = ugraph,
        op_attr = dict(),
        input_tensors = [],
        output_tensors = []
    )
    bias_op.op_attr["value"] = np.array([1], dtype=np.int8)
    bias_op.op_attr["shape"] = [1]

    bias = TensorInfo(
        name = "bias",
        op_name = "bias_const",
        dtype = np.dtype("int8"),
        shape = bias_op.op_attr["shape"],
        ugraph = ugraph
    )
    bias_op.output_tensors = [bias]


    fc1_op = OperationInfo(
        name = "FC1",
        op_type = "FULLY_CONNECTED",
        backend = "tensorflow",
        ugraph = ugraph,
        op_attr = dict(),
        input_tensors = [],
        output_tensors = []
    )

    output = TensorInfo(
        name = "output",
        op_name = "FC1",
        dtype = np.dtype("float"),
        shape = [1, 1],
        ugraph = ugraph
    )

    fc1_op.input_tensors = [input1, weight, bias]
    fc1_op.output_tensors = [output]

    ugraph.ops_info["FC1"] = fc1_op
    ugraph.output_nodes = ["FC1"]
    #ugraph.backend = "tensorflow"

    topologic_order_graph(ugraph)
    #ugraph = prune_graph(ugraph)

    return ugraph
