# -*- coding:utf8 -*-
r"""Graph Visualization Transformer

Transformers that outputs a image file presenting the ugraph before the transformation
"""
import re
from collections import defaultdict
from copy import deepcopy

from utensor_cgen.ir import OperationInfo, uTensorGraph
from utensor_cgen.utils import parse_tensor_name

from .base import Transformer

__all__ = ["GraphVizTransformer"]


class GraphVizTransformer(Transformer):
  METHOD_NAME = 'graphViz'
  KWARGS_NAMESCOPE = '_utensor_graphViz'
  
  def transform(self, ugraph):
    ugraph.viz_graph(fname="output.gv")

    return ugraph