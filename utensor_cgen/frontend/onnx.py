import onnx
from onnx_tf.backend import prepare

from utensor_cgen.frontend.base import Parser
from utensor_cgen.frontend import FrontendSelector
from .tensorflow import GraphDefParser


@FrontendSelector.register(target_exts=['.onnx'])
class OnnxParser(Parser):

  @classmethod
  def parse(cls, onnx_file, output_nodes):
    onnx_model = onnx.load(onnx_file)
    tf_rep = prepare(onnx_model)
    graph_def = tf_rep.graph.as_graph_def()
    ugraph = GraphDefParser.parse(graph_def, output_nodes)
    return ugraph