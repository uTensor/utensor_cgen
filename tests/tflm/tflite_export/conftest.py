import numpy as np
from pytest import fixture

import tensorflow as tf
from utensor_cgen.ir import TensorInfo, OperationInfo, uTensorGraph
from utensor_cgen.utils import prune_graph, topologic_order_graph

@fixture(name='hybrid_quant_output')
def simple_tflm_graph():
    ugraph = uTensorGraph()

    mock_input_op0 = OperationInfo(
        name = "mock_input_const0",
        op_type = "Const",
        backend = "tensorflow",
        ugraph = ugraph,
        op_attr = dict(),
        input_tensors = [],
        output_tensors = []
    )
    mock_input_op0.op_attr["value"] = np.array([[2],[4],[6],[8]], dtype=np.float32)
    mock_input_op0.op_attr["shape"] = [4,1]

    input0 = TensorInfo(
        name = "input0",
        op_name = "mock_input_const0",
        dtype = mock_input_op0.op_attr["value"].dtype,
        shape = mock_input_op0.op_attr["shape"],
        ugraph = ugraph
    )

    mock_input_op0.output_tensors = [input0]

    mock_input1_op = OperationInfo(
        name = "mock_input_const1",
        op_type = "Const",
        backend = "tensorflow",
        ugraph = ugraph,
        op_attr = dict(),
        input_tensors = [],
        output_tensors = []
    )
    mock_input1_op.op_attr["value"] = np.array([[2],[4],[6],[8]], dtype=np.float32)
    mock_input1_op.op_attr["shape"] = [4,1]

    input1 = TensorInfo(
        name = "input1",
        op_name = "mock_input_const1",
        dtype = mock_input1_op.op_attr["value"].dtype,
        shape = mock_input1_op.op_attr["shape"],
        ugraph = ugraph
    )

    mock_input1_op.output_tensors = [input1]

    add_output = TensorInfo(
        name = "add_out",
        op_name = "add0",
        dtype = mock_input_op0.op_attr["value"].dtype,
        shape = mock_input_op0.op_attr["shape"],
        ugraph = ugraph
    )

    add_op = OperationInfo(
        name = "add0",
        op_type = "ADD",
        backend = "tensorflow",
        ugraph = ugraph,
        op_attr = dict(),
        input_tensors = [input0, input1],
        output_tensors = [add_output]
    )

    ugraph.ops_info["ADD0"] = add_op

    weight_op = OperationInfo(
        name = "weight_const",
        op_type = "Const",
        backend = "tensorflow",
        ugraph = ugraph,
        op_attr = dict(),
        input_tensors = [],
        output_tensors = []
    )
    #weight_op.op_attr["value"] = np.array([1,2,3,4], dtype=np.int8)
    weight_op.op_attr["value"] = np.array([10,20,30,40], dtype=np.float32)
    weight_op.op_attr["shape"] = [1,4]

    weight = TensorInfo(
        name = "weight",
        op_name = "weight_const",
        dtype = np.dtype("float32"),
        shape = weight_op.op_attr["shape"],
        ugraph = ugraph
    )
    weight_op.output_tensors = [weight]

    bias_op = OperationInfo(
        name = "bias_const",
        op_type = "Const",
        backend = "tensorflow",
        ugraph = ugraph,
        op_attr = dict(),
        input_tensors = [],
        output_tensors = []
    )
    #bias_op.op_attr["value"] = np.array([1], dtype=np.int8)
    bias_op.op_attr["value"] = np.array([7], dtype=np.float32)
    bias_op.op_attr["shape"] = [1]

    bias = TensorInfo(
        name = "bias",
        op_name = "bias_const",
        dtype = np.dtype("float32"),
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
        dtype = np.dtype("float32"),
        shape = [1],
        ugraph = ugraph
    )

    fc1_op.input_tensors = [add_output, weight, bias]
    fc1_op.output_tensors = [output]

    ugraph.ops_info["FC1"] = fc1_op
    ugraph.output_nodes = ["FC1"]
    #ugraph.backend = "tensorflow"

    topologic_order_graph(ugraph)
    #ugraph = prune_graph(ugraph)

    #return: ugraph, input tensors, output tensors
    return [ugraph, ["input0", "input1"], ["weight", "bias", "output"]]
