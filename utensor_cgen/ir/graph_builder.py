class uTensorGraphBuilderMixin(object):

  def add_op(self, op_type, *args, **kwargs):
    from utensor_cgen.backend.operators import OperatorFactory
    op_info = OperatorFactory.build_op_info(
      self,
      op_type,
      *args,
      **kwargs
    )
    return op_info.output_tensors
