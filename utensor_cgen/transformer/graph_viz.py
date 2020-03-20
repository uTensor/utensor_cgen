# -*- coding:utf8 -*-
r"""Graph Visualization Transformer

Transformers that outputs a image file presenting the ugraph before the transformation
"""
from utensor_cgen.ir.misc.graph_viz import viz_graph

from .base import Transformer
from .pipeline import TransformerPipeline

__all__ = ["GraphVizTransformer"]


@TransformerPipeline.register_transformer
class GraphVizTransformer(Transformer):
  METHOD_NAME = 'graph_viz'
  KWARGS_NAMESCOPE = '_utensor_graph_viz'

  def __init__(self, out_fname="graph.gv", view=False, cleanup=True):
    self.out_fname = out_fname
    self.view = view
    self.prune_graph = False
    self.cleanup = cleanup
  
  def transform(self, ugraph):
    viz_graph(
      ugraph=ugraph,
      out_fname=self.out_fname,
      view=self.view,
      cleanup=self.cleanup
    )
    return ugraph
