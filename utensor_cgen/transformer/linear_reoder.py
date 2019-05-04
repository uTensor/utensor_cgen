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
from utensor_cgen.ir.misc.graph_viz import viz_graph

from .base import Transformer

__all__ = ["CMSIS_NN_Transformer"]

class Linear_Reorder_Transformer(Transformer):
  METHOD_NAME = 'linear_reoder'
  KWARGS_NAMESCOPE = '_linear_reoder'

  def get_matcher_graph(self):
    ugraph = uTensorGraph(output_nodes=['maxpool'])

    dummpy_input0 = Const_Op('dummy_input0', np.zeros([16,16]), ugraph)
    dummpy_input1 = Const_Op('dummy_input1', np.zeros([4,4]), ugraph)
    conv_out = conv2d_op('convolution2d', [dummpy_input0[0], dummpy_input1[0]], ugraph)
    relu_out = relu_op('relu', conv_out, ugraph)
    out_tensor = maxpool_op('maxpool', relu_out, ugraph)
    topologic_order_graph(ugraph)

    #viz_graph('matcher', True, ugraph)
    
    meta = dict()
    meta["dummy_input0"] = ["End", "Any"]
    meta["dummy_input1"] = ["End", "Any"]
    
    return (ugraph, meta)

  def transform(self, ugraph):
    [matcher_ugraph, metaData] = self.get_matcher_graph()
    while True:
      matcher = uGraphMatcher()
      result = matcher.isomorphic_match(ugraph, matcher_ugraph, metaData)
      if result == False:
        break

      max_pool_op = matcher['maxpool']
      relu_op = matcher['relu']

      max_pool_op.input_tensors[0] = matcher['convolution2d:0']
      max_pool_op.output_tensors[0] = matcher['relu:0']
      relu_op.input_tensors[0] = matcher['relu:0']
      relu_op.output_tensors[0] = matcher['maxpool:0']

      matcher['maxpool'] = max_pool_op
      matcher['relu'] = relu_op

      update_tensor_op_names(ugraph)
      graph_validate(ugraph)
      topologic_order_graph(ugraph)
      #import pdb; pdb.set_trace()

      viz_graph('matcher', True, ugraph)
    return ugraph ##remove me

    # graph_check(ugraph)
    # return ugraph
