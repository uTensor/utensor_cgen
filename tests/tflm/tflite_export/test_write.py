import numpy as np

import tensorflow as tf
from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.matcher import uTensorGraphMatcher
from utensor_cgen.utils import prune_graph, topologic_order_graph
from utensor_cgen.transformer import TFLiteExporter
import flatbuffers
import utensor_cgen.third_party.tflite as tflite
from utensor_cgen.third_party.tflite import *
from utensor_cgen.third_party.tflite.BuiltinOperator import BuiltinOperator
from utensor_cgen.third_party.tflite.Model import Model
from utensor_cgen.third_party.tflite.BuiltinOptions import BuiltinOptions
from utensor_cgen.third_party.tflite.TensorType import TensorType

builtin_ops = {v: k for k, v in BuiltinOperator.__dict__.items()}
op_options = {v: k for k, v in BuiltinOptions.__dict__.items()}

tensor_np_type = dict()
tensor_np_type[0] = np.float32
tensor_np_type[1] = np.float16
tensor_np_type[2] = np.int32
tensor_np_type[3] = np.uint8
tensor_np_type[4] = np.uint64
tensor_np_type[5] = np.ubyte #FIXME: supposed to be string
tensor_np_type[6] = np.bool
tensor_np_type[7] = np.int16
tensor_np_type[8] = np.cdouble
tensor_np_type[9] = np.int8

def print_tflite_graph(byte_buff):

    model = Model.GetRootAsModel(byte_buff, 0)
    subgraphs_len = model.SubgraphsLength()
    subgraph = model.Subgraphs(0)
    n_ops = subgraph.OperatorsLength()
    print("version: ", model.Version())
    print("subgraph len: ", subgraphs_len)
    print("number of operators: ", n_ops)
    print("number of t buff: ", model.BuffersLength())
    print("flat buffer length: ", len(byte_buff), " bytes")
    op_codes = []
    for i in range(0, model.OperatorCodesLength()):
        op_code =  model.OperatorCodes(i)
        op_codes.append(op_code)
    print("op code length: ", len(op_codes))

    for i in range(0, subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        print("op code index: ", op.OpcodeIndex())
        opIndex = op.OpcodeIndex()
        op_code = op_codes[opIndex]
        builtin_code = op_code.BuiltinCode()
        op_type = builtin_ops[builtin_code]
        print(op_type)
        
        input_tensors = [subgraph.Tensors(input_idx) for input_idx in op.InputsAsNumpy()]
        for tensor in input_tensors:
            print()
            print(tensor.Name(), ", ", tensor.ShapeAsNumpy())
            print("variable: ", tensor.IsVariable())
            if tensor.Type() == np.uint8 or tensor.Type() == np.int8:
                q = tensor.Quantization()
                assert q != None
                print("quantization info: ")
                print(" Detail Type: ", q.DetailsType())
                print("      Scales: ", q.ScaleAsNumpy())
                print("       Zeros: ", q.ZeroPointAsNumpy())
                print("       Scale: ", q.ScaleAsNumpy())
                print("  Zero Point: ", q.ZeroPointAsNumpy())      
                print("   Dimension: ", q.QuantizedDimension())      
            
            print(tensor.IsVariable())
            if not tensor.IsVariable():
                buffer_index = tensor.Buffer()
                assert buffer_index >= 0
                assert model.Buffers(buffer_index).DataLength() > 0
                buffer_content = model.Buffers(buffer_index).DataAsNumpy()
                print("Tensor values: ", buffer_content.astype(tensor_np_type[tensor.Type()]))
            else:
                print("None")

def test_tflite_fb_write(hybrid_quant_output):
    [sample_ugraph, input_tensors, output_tensors] = hybrid_quant_output
    exporter = TFLiteExporter(input_tensors=input_tensors, output_tensors=output_tensors)
    ugraph = exporter.transform(sample_ugraph)
    model_content = exporter.output()

    print_tflite_graph(model_content)
    
    # referece_model_content = open('/Users/neitan01/Documents/tflm/sinExample/sine_model.tflite', "rb").read()
    # print_tflite_graph(referece_model_content)

    open("tflm_test_model.tflite", "wb").write(model_content)
    test_model = tf.lite.Interpreter('tflm_test_model.tflite')
    test_model.allocate_tensors()
    test_model.invoke()

    print("0 :", test_model.get_tensor(0))
    print("1 :", test_model.get_tensor(1))
    print("2 :", test_model.get_tensor(2))
    print("3 :", test_model.get_tensor(3))


    print(test_model.get_tensor_details())
    print("out0 :", test_model.get_tensor(test_model.get_output_details()[0]["index"]))
    print("out1 :", test_model.get_tensor(test_model.get_output_details()[1]["index"]))
    print("out2 :", test_model.get_tensor(test_model.get_output_details()[2]["index"]))


    test_pass = True
    assert test_pass, 'error message here'
