import inspect
from contextlib import contextmanager

import numpy as np

from utensor_cgen.utils import LazyAttrib, LazyLoader, topologic_order_graph

from .converter import (AttrValueConverter, DataTypeConverter,
                        GenericTensorConverterMixin)

# lazy loading for fixing cyclic import
operators = LazyLoader(submod_name='backend.utensor.code_generator.rearch._operators')
ir_base = LazyLoader(submod_name='ir.base')
OperationInfo = LazyAttrib(ir_base, 'OperationInfo')
TensorInfo = LazyAttrib(ir_base, 'TensorInfo')


class GraphFinalizedError(Exception): pass


# uTensorGraphBuilderMixin is solely for ir.uTensorGraph.
# Don't mix it with any other class
class uTensorGraphBuilderMixin(object):

  def add_op(self, op_type, *args, name=None, is_output=False, **kwargs):
    """
    Add operator to the graph and return the output tensors of the operator

    :param op_type: the operation type to add, ex: AddOperator
    :type op_type: string

    :param *args: the input tensors
    :type *args: :class:`utensor.ir.base.TensorInfo`

    :param name: the name of the operation. Auto-generate if not given
    :type name: string

    :param is_output: if set true, the operation will be added as graph's output nodes
    :type is_output: bool

    :param **kwargs: other keyword arguments to be passed to operator's construction callback
    :type **kwargs: dict

    :rtype: List[:class:`utensor.ir.base.TensorInfo`]
    """
    if self.is_finalized:
      raise GraphFinalizedError(
        'Can not add op to finalized graph'
      )
    if name is None:
      n = len(self.get_ops_by_type(op_type)) + 1
      name = f'{op_type}_{n}'
    if name in self.ops_info:
      raise ValueError('Duplicate op_name: %s' % name)
    if op_type == "Placeholder":
      op_info = self._build_placeholder_op_info(
        op_name=name,
        shape=kwargs['shape'],
        dtype=kwargs.get('dtype', np.dtype('float32'))
      )
    elif op_type == "Constant":
      op_info = self._build_const_op_info(
        op_name=name,
        values=kwargs['values']
      )
    else:
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

  def _build_const_op_info(self, op_name, values: np.ndarray):
    out_tensor = TensorInfo(
      name=f"{op_name}:0",
      op_name=op_name,
      dtype=values.dtype,
      shape=list(values.shape),
      ugraph=self,
    )
    generic_values = GenericTensorConverterMixin.__utensor_generic_type__(
      np_array=values
    )
    return OperationInfo(
      name=op_name,
      op_type='Constant',
      lib_name=self.lib_name,
      ugraph=self,
      input_tensors=[],
      output_tensors=[out_tensor],
      op_attr={
        'value': AttrValueConverter.__utensor_generic_type__(
          value_name='tensor', value=generic_values
        ),
        'dtype': AttrValueConverter.__utensor_generic_type__(
          value_name='type', value=DataTypeConverter.get_tf_value(values.dtype)
        )
      }
    )

  def _build_placeholder_op_info(self, op_name, shape, dtype):
    out_tensor = TensorInfo(
      name=f"{op_name}:0",
      op_name=op_name,
      dtype=dtype,
      shape=shape,
      ugraph=self,
    )
    return OperationInfo(
      name=op_name,
      lib_name=self.lib_name,
      ugraph=self,
      input_tensors=[],
      output_tensors=[out_tensor],
      op_type="Placeholder",
    )
