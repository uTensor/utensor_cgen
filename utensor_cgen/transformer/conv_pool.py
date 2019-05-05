# -*- coding:utf8 -*-
r"""Convolution Maxpool Fusion Transformer

Node fusion for QuantConv2d QuantMaxPool operators

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
from utensor_cgen.ir.misc.graph_viz import viz_graph

from .base import Transformer

__all__ = ["CONV_POOL_Transformer"]

class CONV_POOL_Transformer(Transformer):
  METHOD_NAME = 'conv_pool'
  KWARGS_NAMESCOPE = '_conv_pool'

  def get_matcher_graph(self):
    ugraph = uTensorGraph(output_nodes=['quantized_maxpool'], backend="tensorflow")

    dummpy_input0 = Const_Op('dummy_input0', np.zeros([16,16], dtype=np.uint8), ugraph)
    dummpy_input0_min = Const_Op('dummy_input0_min', np.zeros([1]), ugraph)
    dummpy_input0_max = Const_Op('dummy_input0_max', np.zeros([1]), ugraph)

    dummpy_input1 = Const_Op('dummy_input1', np.zeros([4,4]), ugraph)
    dummpy_input1_min = Const_Op('dummy_input1_min', np.zeros([1]), ugraph)
    dummpy_input1_max = Const_Op('dummy_input1_max', np.zeros([1]), ugraph)

    conv_out = quantized_conv2d_op('convolution2d', [dummpy_input0[0],
                dummpy_input1[0], dummpy_input0_min[0], dummpy_input0_max[0],
                dummpy_input1_min[0], dummpy_input1_max[0]], ugraph)

    requantization_range_out = requantization_range_op('requantization_range', conv_out, ugraph)

    requantize_out = requantize_op('requantize', [conv_out[0], conv_out[1], conv_out[2],
                     requantization_range_out[0], requantization_range_out[1]], ugraph) #FIXME: check the tensor ordering here

    quantized_maxpool_op('quantized_maxpool', requantize_out, ugraph)

    topologic_order_graph(ugraph)

    #viz_graph('matcher_quant', True, ugraph)
    #import pdb; pdb.set_trace()
    
    meta = dict()
    meta["convolution2d"] = ["End"]
    
    return (ugraph, meta)

  def transform(self, ugraph):
    [matcher_ugraph, metaData] = self.get_matcher_graph()
    while True:
      matcher = uGraphMatcher()
      result = matcher.isomorphic_match(ugraph, matcher_ugraph, metaData)
      if result == False:
        break
    
      import pdb; pdb.set_trace() #remove me
      return ugraph  #remove me

      #swapping the ops
      max_pool_op = matcher['maxpool']
      relu_op = matcher['relu']

      max_pool_op.input_tensors[0] = matcher['convolution2d:0']
      max_pool_op.output_tensors[0] = matcher['relu:0']
      relu_op.input_tensors[0] = matcher['relu:0']
      relu_op.output_tensors[0] = matcher['maxpool:0']

      matcher['maxpool'] = max_pool_op
      matcher['relu'] = relu_op

      #swapping the tensor names
      relu_tensor_name = matcher['relu:0'].name
      maxpool_tensor_name = matcher['maxpool:0'].name

      rename_tensor(relu_tensor_name, 'tmp_relu_name', ugraph)
      rename_tensor(maxpool_tensor_name, relu_tensor_name, ugraph)
      rename_tensor('tmp_relu_name', maxpool_tensor_name, ugraph)
      
      update_tensor_op_names(ugraph)
      graph_validate(ugraph)
      topologic_order_graph(ugraph)
      #import pdb; pdb.set_trace()

      viz_graph('matcher', True, ugraph)
    return ugraph ##remove me

    # graph_check(ugraph)
    # return ugraph
