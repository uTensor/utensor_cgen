from tensorflow.tools.graph_transforms import TransformGraph

from utensor_cgen.ir.base import uTensorGraph
from .base import Transformer

__all__ = ['QuantizeTransformer']

class QuantizeTransformer(Transformer):

  METHOD_NAME = 'quantize'
  KWARGS_NAMESCOPE = '_quantize'

  def transform(self, ugraph):
    graph_def = ugraph.graph_def
    quant_graph_def = TransformGraph(input_graph_def=graph_def,
                                     inputs=[],
                                     outputs=ugraph.output_nodes,
                                     transforms=["quantize_weights", "quantize_nodes"])
    return uTensorGraph(graph=quant_graph_def, output_nodes=ugraph.output_nodes)
