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

    conv_out = quantized_conv2d_op('quant_convolution2d', [dummpy_input0[0],
                dummpy_input1[0], dummpy_input0_min[0], dummpy_input0_max[0],
                dummpy_input1_min[0], dummpy_input1_max[0]], ugraph)

    requantization_range_out = requantization_range_op('requantization_range', conv_out, ugraph)

    requantize_out = requantize_op('requantize', [conv_out[0], conv_out[1], conv_out[2],
                     requantization_range_out[0], requantization_range_out[1]], ugraph) #FIXME: check the tensor ordering here

    quantized_maxpool_op('quantized_maxpool', requantize_out, ugraph)

    topologic_order_graph(ugraph)

    
    meta = dict()
    meta["quant_convolution2d"] = ["End"]
    
    return (ugraph, meta)

  def transform(self, ugraph):
    [matcher_ugraph, metaData] = self.get_matcher_graph()
    while True:
      matcher = uGraphMatcher()
      result = matcher.isomorphic_match(ugraph, matcher_ugraph, metaData)
      if result == False:
        break

      fused_op_name = matcher['quant_convolution2d'].name + "_" + matcher['quantized_maxpool'].name
      fused_op_out = quantized_conv2d_pool_op(fused_op_name, matcher['quant_convolution2d'].input_tensors, ugraph)
      matcher['quantized_maxpool:0'] = fused_op_out[0]
      matcher['quantized_maxpool:1'] = fused_op_out[1]
      matcher['quantized_maxpool:2'] = fused_op_out[2]
      matcher['quant_convolution2d'] = None
      matcher['requantization_range'] = None
      matcher['requantize'] = None
      matcher['quantized_maxpool'] = None
      
      update_tensor_op_names(ugraph)
      topologic_order_graph(ugraph)
      graph_validate(ugraph)

    
    viz_graph('matcher', True, ugraph)
    import pdb; pdb.set_trace()

    return ugraph ##remove me
