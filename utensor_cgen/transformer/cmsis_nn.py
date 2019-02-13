# -*- coding:utf8 -*-
r"""CMSIS-NN Transformer

Node fusion and replacement for CMSIS-NN

"""
import re
from collections import defaultdict
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

from utensor_cgen.experimental.ugraph_builder import *
from utensor_cgen.experimental.ugraph_matcher import *
from utensor_cgen.experimental.ugraph_util_functions import *
from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.ir import OperationInfo, TensorInfo, uTensorGraph
from utensor_cgen.ir.converter import AttrValueConverter  # hue hue hue hue hue
from utensor_cgen.ir.converter import GenericTensorConverterMixin
from utensor_cgen.ir.utils import graph_check
from utensor_cgen.utils import parse_tensor_name, topologic_order_graph

from .base import Transformer

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
      mgraph = GraphDefParser.parse(quant_graph_def, output_nodes=['zscore/eightbit'])

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

      #turn v * M into M * v
      #pM = transpose_offline(matcher["weight_quantized_const"])
      pM = transpose_offline(matcher["weight_quantized_const"])
      matcher["weight_quantized_const"] = pM
      matcher["weight_quantized_const:0"] = pM.output_tensors[0]

      #turn matmal_eightbit/input/quantize:0 from [1 n] to [n 1]
      pV = matcher["matmal_eightbit/input/quantize"]
      act_reshape_shape = pV.output_tensors[0].shape[::-1]

### reshape
      act_transpose_op_name = pV.name + "_transpose"
      act_transposed_tensors = Const_Reshape(act_transpose_op_name, [pV.output_tensors[0]], act_reshape_shape, ugraph)

      ## convert the inputs Uint8Q7OriginOp
      new_input0_op_name = "convert_uint8_q7_" + act_transposed_tensors[0].name.replace(":", "_")  #pV

      input0_q7_out = Uint8Q7Origin_Op(new_input0_op_name,
                                     [act_transposed_tensors[0],
                                      matcher["matmal_eightbit/input/quantize:1"],
                                      matcher["matmal_eightbit/input/quantize:2"]],
                                      ugraph)

      new_input1_op_name = "convert_uint8_q7_" + matcher["weight_quantized_const"].name  #pM

      input1_q7_out = Uint8Q7Origin_Op(new_input1_op_name,
                                       [matcher["weight_quantized_const:0"],
                                       matcher["weight_quantized_min:0"],
                                       matcher["weight_quantized_max:0"]],
                                       ugraph)

      #using CMSIS-NN FC as MatMul only, for now
      #generate new op name
      new_op_name = "cmsis_fc_" + matcher["matmal/eightbit"].name

      #bias
      bias_name = new_op_name + "_bias"
      #FIXME: for debugging purpose, temporarily fixing the bias values to 0
      bias_values = np.full(act_reshape_shape, 0)

      bias_out_tensors = Const_Op(bias_name + "_bias", bias_values, ugraph)

      #bias shift
      bShift_tensors = Const_Op(matcher["matmal/eightbit"].name + "_bShift", np.array([0], dtype=np.uint16), ugraph)

      oShift_tensors = Const_Op(matcher["matmal/eightbit"].name + "_oShift", np.array([0], dtype=np.uint16), ugraph)

      scratch_space = "cmsis_scratch_" + matcher["matmal/eightbit"].name
      scratch_shape = list(map(lambda x: x if x else 1, matcher['matmal_eightbit/input/quantize:0'].shape))
      scratch_tensors = Ram_Op(scratch_space, np.zeros(tuple(scratch_shape), dtype=np.uint16), ugraph)
    
      new_op_name = "cmsis_fc_" + matcher["matmal/eightbit"].name

      cmsis_fc_out = CMSIS_FC_Op(new_op_name, input0_q7_out, input1_q7_out,
                  bias_out_tensors, bShift_tensors, oShift_tensors,
                  scratch_tensors, ugraph)

      # ugraph.drop_op(result[0]['matmal/eightbit/requant_range'])
      # ugraph.drop_op(result[0]['matmal/eightbit/requantize'])
      # ugraph.drop_op(result[0]['zscore/eightbit'])
      # ugraph.drop_op(result[0]['zscore/eightbit/requant_range'])
      # ugraph.drop_op(result[0]['zscore/eightbit/requantize'])
      #ugraph.add_op(fused_op_info)

      #output reshape
      act_reshape_op_name = new_op_name + "_newshape"
      matmul_output_shape = list(cmsis_fc_out[0].shape)
      matmul_output_shape.reverse()
      reshape_out = Const_Reshape(act_reshape_op_name, cmsis_fc_out, matmul_output_shape, ugraph)
      matcher["matmal/eightbit:0"] = reshape_out[0]

      #range op
      new_range_op_name = new_op_name + "_range"

      range_out = QuantRangeForMultiplicationu8u8int32_Op(new_range_op_name,
                                                    [matcher["matmal_eightbit/input/quantize:1"], matcher["matmal_eightbit/input/quantize:2"]],
                                                    [matcher["weight_quantized_min:0"], matcher["weight_quantized_max:0"]],
                                                    ugraph)

      matcher["matmal/eightbit:1"] = range_out[0]
      matcher["matmal/eightbit:2"] = range_out[1]

      matcher['matmal/eightbit'] = None

      topologic_order_graph(ugraph)
      graph_validate(ugraph)

    graph_check(ugraph)
    return ugraph
