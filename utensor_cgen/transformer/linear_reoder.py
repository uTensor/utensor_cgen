# -*- coding:utf8 -*-
r"""Linear Re-ordering Transformer

Linear Operation Legalizations

"""
import logging

import tensorflow as tf
from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.matcher import uTensorGraphMatcher
from utensor_cgen.utils import prune_graph, topologic_order_graph

from .base import Transformer
from .pipeline import TransformerPipeline

__all__ = ["LinearReorderTransformerV2"]
logger = logging.getLogger(__name__)


@TransformerPipeline.register_transformer
class LinearReorderTransformerV2(Transformer):
  METHOD_NAME = 'linear_reorder'
  KWARGS_NAMESCOPE = '_linear_reorder'

  def __init__(self, max_retry=10):
    self.max_retry = max_retry
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
    matches = matcher.match_all(ugraph)
    retry_cnt = 0
    while matches and retry_cnt < self.max_retry:
      match = matches[0]
      subj_relu = match.patrn2subj_op_map['relu']
      if len(subj_relu.output_nodes) == 1:
        """If the relu op is not consume only by one pool op, skip.
        (No trivial way to determine how to resolve the execution order of ops)
        """
        ugraph = match.replace_with(callback=self)
      retry_cnt += 1
      matches = matcher.match_all(ugraph)
    if retry_cnt == self.max_retry and matches:
      logger.warning(
        'max retry %s reached, there are still matches found in the graph: %s',
        self.max_retry,
        matches,
      )
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
    topologic_order_graph(ugraph)
    ugraph = prune_graph(ugraph)
    input_map = {
      match.pattern_ugraph['relu'].input_tensors[0]:ugraph['max_pool'].input_tensors[0]
    }
    output_map = {
      match.pattern_ugraph['max_pool'].output_tensors[0]:ugraph['relu'].output_tensors[0]
    }
    return ugraph, input_map, output_map
