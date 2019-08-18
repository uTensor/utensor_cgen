import inspect
from contextlib import contextmanager


class GraphFinalizedError(Exception): pass


class uTensorGraphBuilderMixin(object):

  def add_op(self, *args, op_type, name, is_output=False, **kwargs):
    # FIXME: cyclic imports... OMG
    from utensor_cgen.backend.operators import OperatorFactory
    if self.is_finalized:
      raise GraphFinalizedError(
        'Can not add op to finalized graph'
      )
    if name in self.ops_info:
      raise ValueError('Duplicate op_name: %s' % name)
    op_info = OperatorFactory.build_op_info(
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
    from utensor_cgen.backend.operators import OperatorFactory
    op_cls = OperatorFactory.get_opertor(op_type)
    return inspect.signature(op_cls.build_op_info)

  def finalize(self):
    from utensor_cgen.utils import topologic_order_graph
    self._is_finalized = True
    topologic_order_graph(self)

  @property
  def is_finalized(self):
    if not hasattr(self, '_is_finalized'):
      self._is_finalized = False
    return self._is_finalized

  @staticmethod
  def list_all_ops():
    from utensor_cgen.backend.operators import OperatorFactory
    return list(OperatorFactory._operators.keys())

  @contextmanager
  def begin_construction(self):
    if self.is_finalized:
      raise GraphFinalizedError('this graph is finalized, no construction allowed')
    yield
    self.finalize()
