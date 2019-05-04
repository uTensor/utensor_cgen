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
      #import pdb; pdb.set_trace()
      relu_name = matcher['relu'].name + '_'
      maxpool_name = matcher['maxpool'].name + '_'

      new_relu_out = relu_op(relu_name, [matcher['relu:0']], ugraph)
      new_maxpool_out = maxpool_op(maxpool_name, [matcher['convolution2d:0']], ugraph)
      matcher['relu:0'] = new_maxpool_out[0]
      matcher['maxpool:0'] = new_relu_out[0]
      matcher['relu'] = None
      matcher['maxpool'] = None

      topologic_order_graph(ugraph)
      graph_validate(ugraph)

      viz_graph('matcher', True, ugraph)
    return ugraph ##remove me

    # graph_check(ugraph)
    # return ugraph
