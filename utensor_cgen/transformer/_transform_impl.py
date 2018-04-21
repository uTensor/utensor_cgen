from abc import ABCMeta, abstractmethod


class _DropoutFuseTransformer(object):

  def transform(self, op_infos, topo_orders, output_nodes):
    pass


class _BatchNormFuseTransformer(object):

  def transform(self, op_infos, topo_orders, output_nodes):
    pass


class Transformer(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def transform(self, op_infos, topo_orders, output_nodes):
    raise NotImplementedError('You should overwrite transform method for all transformer')


class FuseOpTransformer(Transformer):

  _DELEGATION_MAP = {
    "dropout": _DropoutFuseTransformer,
    "BatchNorm": _BatchNormFuseTransformer
  }

  def __init__(self, target_name_scope):
    if target_name_scope not in self._DELEGATION_MAP:
      raise ValueError('Unsupport fuse type: {}'.format(target_name_scope))
    self._target_ns = target_name_scope
    self._delegate = self._DELEGATION_MAP[target_name_scope]

  def transform(self, op_infos, topo_orders, output_nodes):
    new_outputs = self._delegate.transform(op_infos, topo_orders, output_nodes)
    return new_outputs
