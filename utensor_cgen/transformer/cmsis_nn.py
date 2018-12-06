# -*- coding:utf8 -*-
r"""CMSIS-NN Transformer

Node fusion and replacement for CMSIS-NN

"""
import re
from collections import defaultdict
from copy import deepcopy

import tensorflow as tf
import numpy as np

from utensor_cgen.ir import OperationInfo, uTensorGraph, TensorInfo
from utensor_cgen.ir.converter import AttrValueConverter, GenericTensorConverterMixin # hue hue hue hue hue
from utensor_cgen.utils import parse_tensor_name
from tensorflow.tools.graph_transforms import TransformGraph
from utensor_cgen.ir.utils import graph_check

from .base import Transformer
from utensor_cgen.experimental.ugraph_util_functions import *
from utensor_cgen.experimental.ugraph_matcher import *
from utensor_cgen.experimental.ugraph_builder import *

__all__ = ["CMSIS_NN_Transformer"]

## MatMul Only
class CMSIS_NN_Transformer(Transformer):
  METHOD_NAME = 'cmsisnn'
  KWARGS_NAMESCOPE = '_utensor_cmsisnn'

  def make_rand_const(self, shape, name):
    val = np.random.random(shape)
    return tf.convert_to_tensor(val, name=name, dtype=tf.float32)

  def get_matcher_graph(self):
    graph = tf.Graph()
    tf.reset_default_graph() #remove me
    with graph.as_default():
    
      x = tf.placeholder(dtype=tf.float32, name='input')
      #x = self.make_rand_const([1,784], name='input')
      W_fc1 = self.make_rand_const([784, 128], name='weight')
      b_fc1 = self.make_rand_const([128], name='bias')
      matmal = tf.matmul(x, W_fc1, name='matmal')
      a_fc1 = tf.add(matmal, b_fc1, name="zscore")

      meta = dict()
      meta["matmal_eightbit/input/quantize"] = ["End", "Any"]

      quant_graph_def = TransformGraph(input_graph_def=graph.as_graph_def(),
                                     inputs=['input'],
                                     outputs=['zscore'],
                                     transforms=["quantize_weights", "quantize_nodes"])
      mgraph = uTensorGraph(graph=quant_graph_def, output_nodes=['zscore/eightbit'])

    #mgraph.viz_graph(fname="matcher.gv")
    return (mgraph, meta)

  def transform(self, ugraph):
    [matcher_ugraph, metaData] = self.get_matcher_graph()
    #ugraph.viz_graph(fname="subject.gv")

    while True:
      matcher = uGraphMatcher()
      result = matcher.isomorphic_match(ugraph, matcher_ugraph, metaData)
      if result == False:
        break
      
      tmp_ugraph = uTensorGraph()

      #turn v * M into M * v
      pM = transpose_offline(ugraph.ops_info[result[0]["weight_quantized_const"]])
      ugraph.ops_info[result[0]["weight_quantized_const"]] = pM

      #turn matmal_eightbit/input/quantize:0 from [1 n] to [n 1]
      pV = ugraph.ops_info[result[0]["matmal_eightbit/input/quantize"]]
      act_reshape_shape = pV.output_tensors[0].shape[::-1]
      for i, v in enumerate(act_reshape_shape):
        if v == None:
          act_reshape_shape[i] = 1

### reshape
      act_transpose_op_name = pV.name + "_transpose"
      (act_transpose_opInfo, act_transposed_tensors) = create_const_reshape(act_transpose_op_name, [pV.output_tensors[0]], act_reshape_shape, ugraph)
      ugraph.add_op(act_transpose_opInfo)
      #import pdb; pdb.set_trace()

      ## convert the inputs Uint8Q7OriginOp
      new_input0_op_name = "convert_uint8_q7_" + act_transposed_tensors[0].name  #pV
      new_input1_op_name = "convert_uint8_q7_" + result[0]["weight_quantized_const"]  #pM

      input0_q7_out = TensorInfo(name=result[0]["matmal_eightbit/input/quantize"] + "_q7:0",
                        op_name=new_input0_op_name,  #ownership
                        dtype=np.dtype('int8'),
                        shape=act_reshape_shape,
                        ugraph=tmp_ugraph
                        )
      input0_q7_inputs = list()
      input0_q7_inputs.append(tensorInfo_from_name(ugraph, act_transposed_tensors[0].name))
      input0_q7_inputs.append(tensorInfo_from_name(ugraph, result[1]["matmal_eightbit/input/quantize:1"]))
      input0_q7_inputs.append(tensorInfo_from_name(ugraph, result[1]["matmal_eightbit/input/quantize:2"]))
      input0_q7_op_info = OperationInfo(name=new_input0_op_name,
                            input_tensors=input0_q7_inputs,
                            output_tensors=[input0_q7_out],
                            op_type="Uint8Q7OriginOp",
                            backend="tensorflow",
                            ugraph=tmp_ugraph)

      ugraph.add_op(input0_q7_op_info)

      input1_shape = tensorInfo_from_name(ugraph, result[1]["weight_quantized_const:0"]).shape
      input1_q7_out = TensorInfo(name=result[0]["weight_quantized_const"] + "_q7:0",
                        op_name=new_input1_op_name,  #ownership
                        dtype=np.dtype('int8'),
                        shape=input1_shape,
                        ugraph=tmp_ugraph
                        )
      input1_q7_inputs = list()
      input1_q7_inputs.append(tensorInfo_from_name(ugraph, result[1]["weight_quantized_const:0"]))
      input1_q7_inputs.append(tensorInfo_from_name(ugraph, result[1]["weight_quantized_min:0"]))
      input1_q7_inputs.append(tensorInfo_from_name(ugraph, result[1]["weight_quantized_max:0"]))
      input1_q7_op_info = OperationInfo(name=new_input1_op_name,
                            input_tensors=input1_q7_inputs,
                            output_tensors=[input1_q7_out],
                            op_type="Uint8Q7OriginOp",
                            backend="tensorflow",
                            ugraph=tmp_ugraph
                            )
      ugraph.add_op(input1_q7_op_info)

      #using CMSIS-NN FC as MatMul only, for now
      #generate new op name
      new_op_name = "cmsis_fc_" + result[0]["matmal/eightbit"]

      #bias
      bias_name = new_op_name + "_bias"
      #FIXME: for debugging purpose, temporarily fixing the bias values to 0
      bias_values = np.full(act_reshape_shape, 0)

      (bias_op_info, bias_out_tensors) = create_const_op(bias_name + "_bias", bias_values)
      bias_out_tensor_info = bias_out_tensors[0]
      ugraph.add_op(bias_op_info)

      #bias shift
      (bShift_op_info, bShift_tensors) = create_const_op(result[0]["matmal/eightbit"] + "_bShift", np.array([0], dtype=np.uint16))
      ugraph.add_op(bShift_op_info)

      (oShift_op_info, oShift_tensors) = create_const_op(result[0]["matmal/eightbit"] + "_oShift", np.array([0], dtype=np.uint16))
      ugraph.add_op(oShift_op_info)

      scratch_space = "cmsis_scratch_" + result[0]["matmal/eightbit"]
      scratch_shape = list(map(lambda x: x if x else 1, tensorInfo_from_name(ugraph, result[1]['matmal_eightbit/input/quantize:0']).shape))
      (scratch_op_info, scratch_tensors) = create_ram_op(scratch_space, np.zeros(tuple(scratch_shape), dtype=np.uint16))
      ugraph.add_op(scratch_op_info)
      #compile new op's the input list
      in_tensors = list()
      in_tensors.append(input0_q7_out) #pV
      in_tensors.append(input1_q7_out) # Weights pM
      ##Bias is temporary disabled in CMSIS_NN_FC, so give it anything for now
      in_tensors.append(bias_out_tensor_info) # Bias #FIXME: needs to be the right dimension
      in_tensors.append(bShift_tensors[0])
      in_tensors.append(oShift_tensors[0])
      in_tensors.append(scratch_tensors[0])

      #TODO:
      # make sure weight and bias are in the format
      # supply S_TENSOR bShift and S_TENSOR oShift here
      # This can be found by either offline quantization profiling
      # Or, by taking statistic of the weight matrix
      # In uTensor runtime CMSIS Op, convert the number format back to the linear quantized format
      # update _CMSIS_NN_FCOperator in operator.py

      #compile new op's output list
      subject_matmul_tensors = ugraph.ops_info[result[0]["matmal/eightbit"]].output_tensors
      new_op_name = "cmsis_fc_" + result[0]["matmal/eightbit"]

      matcher_matmul_tensor = TensorInfo(name=new_op_name + ":0",
                        op_name=new_op_name,
                        dtype=subject_matmul_tensors[0].dtype,
                        shape=subject_matmul_tensors[0].shape,
                        ugraph=tmp_ugraph
                        )

      #import pdb; pdb.set_trace()

      #FIXME: shouldn't be Tensorflow backend
      fused_op_info = OperationInfo(name=new_op_name,
                              input_tensors=in_tensors,
                              output_tensors=[matcher_matmul_tensor],
                              op_type="CMSIS_NN_FC",
                              backend="tensorflow",
                              ugraph=tmp_ugraph
                              )

      ugraph = replace_tensors_op(result[0]['matmal/eightbit'], new_op_name, ugraph)
      ugraph.drop_op(result[0]['matmal/eightbit'])
      # ugraph.drop_op(result[0]['matmal/eightbit/requant_range'])
      # ugraph.drop_op(result[0]['matmal/eightbit/requantize'])
      # ugraph.drop_op(result[0]['zscore/eightbit'])
      # ugraph.drop_op(result[0]['zscore/eightbit/requant_range'])
      # ugraph.drop_op(result[0]['zscore/eightbit/requantize'])
      ugraph.add_op(fused_op_info)

      #output reshape

### reshape shape const
      matmul_output_shape = subject_matmul_tensors[0].shape
      for i, v in enumerate(matmul_output_shape):
        if v == None:
          matmul_output_shape[i] = 1
### reshape shape const
      act_reshape_op_name = fused_op_info.name + "_newshape"
      (act_reshape_const_opInfo, act_reshape_const_tensors) = create_const_op(act_reshape_op_name, np.array(matmul_output_shape))
      ugraph.add_op(act_reshape_const_opInfo)

### reshape
      act_transpose_op_name = fused_op_info.name + "_transpose"

      ugraph = replace_tensor_op_by_name(subject_matmul_tensors[0].name, act_transpose_op_name, ugraph)
      act_transpose_opInfo = OperationInfo(name=act_transpose_op_name,
                              input_tensors=[fused_op_info.output_tensors[0], act_reshape_const_tensors[0]],
                              output_tensors=[tensorInfo_from_name(ugraph, subject_matmul_tensors[0].name)],
                              op_type="Reshape",
                              backend="tensorflow",
                              ugraph=tmp_ugraph
                              )
      ugraph.add_op(act_transpose_opInfo)


      #range op
      new_range_op_name = new_op_name + "_range"
      # matmul_min_tensor = TensorInfo(name=new_op_name + ":1",
      #                         op_name=new_range_op_name,
      #                         dtype=np.dtype('float'),
      #                         shape=[1],
      #                         ugraph=tmp_ugraph
      #                         )
      # matmul_max_tensor = TensorInfo(name=new_op_name + ":2",
      #                         op_name=new_range_op_name,
      #                         dtype=np.dtype('float'),
      #                         shape=[1],
      #                         ugraph=tmp_ugraph
      #                         )
      new_range_op_inputs = list()
      new_range_op_inputs.append(tensorInfo_from_name(ugraph, result[1]["matmal_eightbit/input/quantize:1"]))
      new_range_op_inputs.append(tensorInfo_from_name(ugraph, result[1]["matmal_eightbit/input/quantize:2"]))
      new_range_op_inputs.append(tensorInfo_from_name(ugraph, result[1]["weight_quantized_min:0"]))
      new_range_op_inputs.append(tensorInfo_from_name(ugraph, result[1]["weight_quantized_max:0"]))

      new_range_op_outputs = list()
      new_range_op_outputs.append(subject_matmul_tensors[1])
      new_range_op_outputs.append(subject_matmul_tensors[2])
      ugraph = replace_tensor_op_by_name(subject_matmul_tensors[1].name, new_range_op_name, ugraph)
      ugraph = replace_tensor_op_by_name(subject_matmul_tensors[2].name, new_range_op_name, ugraph)
      new_range_op_outputs[0].op_name = new_range_op_name
      new_range_op_outputs[1].op_name = new_range_op_name
      new_range_op_info = OperationInfo(name=new_range_op_name,
                        input_tensors=new_range_op_inputs,
                        output_tensors=new_range_op_outputs,
                        op_type="QuantRangeForMultiplicationu8u8int32Op",
                        backend="tensorflow",
                        ugraph=tmp_ugraph
                        )
      ugraph.add_op(new_range_op_info)
      graph_validate(ugraph)

    graph_check(ugraph)
    ugraph.viz_graph(fname="cmsis_nn.gv")
    return ugraph
