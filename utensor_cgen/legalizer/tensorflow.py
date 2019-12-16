from .base import LegalizerBase


class GraphDefLegalizer(LegalizerBase):
  TARGET = 'tensorflow'

  def legalize_ops(self, ugraph):
    '''Legalize ops to generic ops in given graph
    '''
    if not ugraph.lib_name == self.TARGET:
      raise ValueError(
        'expecting tensorflow graph, get {}'.format(ugraph.lib_name)
      )
    return ugraph

  def legalize_dtype(self, ugraph):
    '''Legalize data types of tensors in given graph
    '''
    if not ugraph.lib_name == self.TARGET:
      raise ValueError(
        'expecting tensorflow graph, get {}'.format(ugraph.lib_name)
      )
    return ugraph
