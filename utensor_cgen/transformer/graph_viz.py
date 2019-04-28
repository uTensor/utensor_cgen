# -*- coding:utf8 -*-
r"""Graph Visualization Transformer

Transformers that outputs a image file presenting the ugraph before the transformation
"""
import re
from collections import defaultdict
from copy import deepcopy

from utensor_cgen.ir import OperationInfo, uTensorGraph
from utensor_cgen.utils import parse_tensor_name
from utensor_cgen.logger import logger

from .base import Transformer

__all__ = ["GraphVizTransformer"]


class GraphVizTransformer(Transformer):
  METHOD_NAME = 'graph_viz'
  KWARGS_NAMESCOPE = '_utensor_graph_viz'

  def __init__(self, out_fname="graph.gv", view=False):
    self.out_fname = out_fname
    self.view = view
  
  def transform(self, ugraph):
    self.viz_graph(ugraph)

    return ugraph
  
  def viz_graph(self, ugraph):
    from graphviz import Digraph
    dot = Digraph()
    nodes = {}
    i = 0
    for node in ugraph.ops:
        nodes[node.name] = chr(ord('a') + i)
        dot.node(nodes[node.name], "%s: %s" % (node.name, node.op_type))
        i += 1
        for n in node.input_tensors:
            if n.name in nodes:
                continue
            nodes[n.name] = chr(ord('a') + i)
            dot.node(nodes[n.name], "%s: Tensor" % n.name)
            i += 1
        for n in node.output_tensors:
            if n.name in nodes:
                continue
            nodes[n.name] = chr(ord('a') + i)
            dot.node(nodes[n.name], "%s: Tensor" % n.name)
            i += 1
    for node in ugraph.ops:
        for n in node.input_tensors:
            dot.edge(nodes[n.name], nodes[node.name])
        for n in node.output_tensors:
            dot.edge(nodes[node.name], nodes[n.name])
    dot.render(self.out_fname, view=self.view)
    logger.info('graph visualization file generated: %s', self.out_fname)