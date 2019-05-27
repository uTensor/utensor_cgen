# -*- coding:utf8 -*-
r"""Linear Re-ordering Transformer

Linear Operation Legalizations

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
from utensor_cgen.matcher import uTensorGraphMatcher
from utensor_cgen.utils import (parse_tensor_name, prune_graph,
                                topologic_order_graph)

from .base import Transformer

__all__ = ["Linear_Reorder_Transformer", "LinearReorderTransformerV2"]

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
      topologic_order_graph(ugraph)
      graph_validate(ugraph)

    return ugraph


class LinearReorderTransformerV2(Transformer):
  METHOD_NAME = 'linear_reorder_v2'
  KWARGS_NAMESCOPE = '_linear_reorder_v2'

  def __init__(self):
    self.prune_graph = False

  @property
  def pattern_ugraph(self):
    graph = tf.Graph()
    with graph.as_default():
      dummy_input = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
      relu = tf.nn.relu(dummy_input, name='relu')
      tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool')
    pattern_ugraph = GraphDefParser.parse(graph.as_graph_def(), output_nodes=['max_pool'])
    pattern_ugraph['relu'].replace_with_null_input_tensor(0)
    pattern_ugraph = prune_graph(pattern_ugraph)
    topologic_order_graph(pattern_ugraph)
    return pattern_ugraph

  def transform(self, ugraph):
    matcher = uTensorGraphMatcher(pattern_ugraph=self.pattern_ugraph)
    matches = matcher.match(ugraph, 1)
    while matches:
      match = matches[0]
      ugraph = match.replace_with(callback=self)
      matches = matcher.match(ugraph, 1)
    return ugraph

  def __call__(self, match):
    graph = tf.Graph()
    subj_pool_name = match.patrn2subj_op_map['max_pool'].name
    subj_pool_op = match.subject_ugraph[subj_pool_name]
    ksize = subj_pool_op.op_attr['ksize'].value.ints_value[:]
    strides = subj_pool_op.op_attr['strides'].value.ints_value[:]
    padding = subj_pool_op.op_attr['padding'].value
    with graph.as_default():
      dummy_input = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
      max_pool = tf.nn.max_pool(dummy_input, ksize=ksize, strides=strides, padding=padding, name='max_pool')
      tf.nn.relu(max_pool, name='relu')
    ugraph = GraphDefParser.parse(graph.as_graph_def(), output_nodes=['relu'])
    ugraph['max_pool'].replace_with_null_input_tensor(0)
    ugraph = prune_graph(ugraph)
    topologic_order_graph(ugraph)
    input_map = {
      match.pattern_ugraph['relu'].input_tensors[0]:ugraph['max_pool'].input_tensors[0]
    }
    output_map = {
      match.pattern_ugraph['max_pool'].output_tensors[0]:ugraph['relu'].output_tensors[0]
    }
    return ugraph, input_map, output_map
