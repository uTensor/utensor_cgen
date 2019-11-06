import inspect
from contextlib import contextmanager

from utensor_cgen.utils import LazyLoader, topologic_order_graph

operators = LazyLoader('backend.operators')


class GraphFinalizedError(Exception): pass


class uTensorGraphBuilderMixin(object):

  def add_op(self, *args, op_type, name, is_output=False, **kwargs):
    if self.is_finalized:
      raise GraphFinalizedError(
        'Can not add op to finalized graph'
      )
    if name in self.ops_info:
      raise ValueError('Duplicate op_name: %s' % name)
    op_info = operators.OperatorFactory.build_op_info(
      *args,
      ugraph=self,
      name=name,
      op_type=op_type,
      **kwargs
    )
    if is_output:
      self.output_nodes.append(op_info.name)
    return op_info.output_tensors

  def get_add_op_signature(self, op_type):
    op_cls = operators.OperatorFactory.get_opertor(op_type)
    return inspect.signature(op_cls.build_op_info)

  def finalize(self):
    self._is_finalized = True
    topologic_order_graph(self)

  @property
  def is_finalized(self):
    if not hasattr(self, '_is_finalized'):
      self._is_finalized = False
    return self._is_finalized

  @staticmethod
  def list_all_ops():
    return list(operators.OperatorFactory._operators.keys())

  @contextmanager
  def begin_construction(self):
    if self.is_finalized:
      raise GraphFinalizedError('this graph is finalized, no construction allowed')
    yield
    self.finalize()
