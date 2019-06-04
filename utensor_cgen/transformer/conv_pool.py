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
from utensor_cgen.ir.converter import (AttrValueConverter,
                                       GenericTensorConverterMixin)
from utensor_cgen.ir.utils import graph_check
from utensor_cgen.matcher import uTensorGraphMatcher
from utensor_cgen.utils import (parse_tensor_name, prune_graph,
                                topologic_order_graph)

from .base import Transformer
from .quantize import QuantizeTransformer

__all__ = ["CONV_POOL_Transformer", "ConvPoolTransformer"]

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
      op_attr = dict()
      op_attr['_utensor_conv'] = matcher['quant_convolution2d'].op_attr
      op_attr['_utensor_pool'] = matcher['quantized_maxpool'].op_attr
      fused_op_out = quantized_conv2d_pool_op(fused_op_name, matcher['quant_convolution2d'].input_tensors, op_attr, ugraph)
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
    return ugraph

class ConvPoolTransformer(Transformer):
  METHOD_NAME = 'conv_pool_v2'
  KWARGS_NAMESCOPE = '_conv_pool_v2'

  def __init__(self):
    super(ConvPoolTransformer, self).__init__(prune_graph=False)

  @property
  def pattern_ugraph(self):
    graph = tf.Graph()
    with graph.as_default():
      dummy_input = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='dummy_input')
      dummy_weight = tf.zeros([32, 32, 3, 10], dtype=tf.float32, name='dummy_weight')
      conv = tf.nn.conv2d(dummy_input, dummy_weight, strides=[1, 2, 2, 1], padding='VALID', name='conv')
      maxpool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='maxpool')
    ugraph = GraphDefParser.parse(graph.as_graph_def(), output_nodes=[maxpool.op.name])
    quant_ugraph = QuantizeTransformer().transform(ugraph)
    patrn_ugraph = deepcopy(quant_ugraph)
    quant_conv_op = patrn_ugraph['conv/eightbit']
    for i, _ in enumerate(quant_conv_op.input_tensors):
      quant_conv_op.replace_with_null_input_tensor(i)
    patrn_ugraph.output_nodes = ['maxpool/eightbit']
    patrn_ugraph = prune_graph(patrn_ugraph)
    topologic_order_graph(patrn_ugraph)
    return patrn_ugraph

  def transform(self, ugraph):
    matcher = uTensorGraphMatcher(pattern_ugraph=self.pattern_ugraph)
    matches = matcher.match(ugraph, n=1)
    while matches:
      match = matches[0]
      ugraph = match.replace_with(callback=self)
      matches = matcher.match(ugraph, n=1)
    return ugraph

  def __call__(self, match):
    op_name = 'quant_conv_pool'
    repl_ugraph = uTensorGraph(
      output_nodes=[op_name],
      backend=match.subject_ugraph.backend
    )
    subj_conv_op = match.patrn2subj_op_map['conv/eightbit']
    subj_pool_op = match.patrn2subj_op_map['maxpool/eightbit']
    output_tensors = [
      TensorInfo(
        name='{}:{}'.format(op_name, i),
        op_name=op_name,
        dtype=subj_tensor.dtype,
        shape=subj_tensor.shape,
        ugraph=repl_ugraph
      )
      for i, subj_tensor in enumerate(subj_pool_op.output_tensors)
    ]
    input_tensors = [
      TensorInfo.make_null_tensor(ugraph=repl_ugraph)
      for _ in subj_conv_op.input_tensors
    ]
    quant_conv2d_pool_op = OperationInfo(
      name=op_name,
      input_tensors=input_tensors,
      n_inputs=len(input_tensors),
      output_tensors=output_tensors,
      n_outputs=len(output_tensors),
      op_type='QuantizedFusedConv2DMaxpool',
      backend=subj_conv_op.backend,
      op_attr={
        '_utensor_conv': subj_conv_op.op_attr,
        '_utensor_pool': subj_pool_op.op_attr,
      },
      ugraph=repl_ugraph
    )
    topologic_order_graph(repl_ugraph)
    input_map = {
      match.pattern_ugraph['conv/eightbit'].input_tensors[0]: quant_conv2d_pool_op.input_tensors[0],
      match.pattern_ugraph['conv/eightbit'].input_tensors[1]: quant_conv2d_pool_op.input_tensors[1],
      match.pattern_ugraph['conv/eightbit'].input_tensors[2]: quant_conv2d_pool_op.input_tensors[2],
      match.pattern_ugraph['conv/eightbit'].input_tensors[3]: quant_conv2d_pool_op.input_tensors[3],
      match.pattern_ugraph['conv/eightbit'].input_tensors[4]: quant_conv2d_pool_op.input_tensors[4],
      match.pattern_ugraph['conv/eightbit'].input_tensors[5]: quant_conv2d_pool_op.input_tensors[5],
    }
    output_map = {
      match.pattern_ugraph['maxpool/eightbit'].output_tensors[0]: output_tensors[0],
      match.pattern_ugraph['maxpool/eightbit'].output_tensors[1]: output_tensors[1],
      match.pattern_ugraph['maxpool/eightbit'].output_tensors[2]: output_tensors[2],
    }
    return repl_ugraph, input_map, output_map
