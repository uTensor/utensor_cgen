class uTensorGraphBuilderMixin(object):

  def add_op(self, op_type, name, *args, **kwargs):
    # FIXME: cyclic imports... OMG
    from utensor_cgen.backend.operators import OperatorFactory
    if self.is_finalized:
      raise RuntimeError(
        'Can not add op to finalized graph'
      )
    if name in self.ops_info:
      raise ValueError('Duplicate op_name: %s' % name)
    op_info = OperatorFactory.build_op_info(
      self,
      op_type,
      name,
      *args,
      **kwargs
    )
    return op_info.output_tensors
  
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
