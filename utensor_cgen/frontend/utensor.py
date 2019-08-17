import pickle

from utensor_cgen.frontend import FrontendSelector
from utensor_cgen.frontend.base import Parser
from utensor_cgen.ir import uTensorGraph
from utensor_cgen.utils import topologic_order_graph


@FrontendSelector.register(target_exts=['.pickle', '.pkl'])
class PickleParser(Parser):

  @classmethod
  def parse(cls, pkl_file, output_nodes=None):
    with open(pkl_file, 'rb') as fid:
      ugraph = pickle.load(fid)
      if not isinstance(ugraph, uTensorGraph):
        raise ValueError('expecting uTensorGraph object, get %s' % type(ugraph))
    if output_nodes is not None:
      ugraph.output_nodes = output_nodes
      topologic_order_graph(ugraph)
    ugraph.finalize()
    return ugraph
