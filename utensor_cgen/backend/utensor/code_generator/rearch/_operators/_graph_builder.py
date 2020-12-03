import numpy as np

from utensor_cgen.ir import OperationInfo, TensorInfo
from utensor_cgen.ir.converter import AttrValueConverter, DataTypeConverter
from utensor_cgen.utils import LazyLoader

from ._base import OperatorFactory, _Operator

tf = LazyLoader('tensorflow.compat.v1')

__all__ = []

def _export(cls):
  __all__.append(cls.__name__)
  return cls


# NOTE: this is recommended pattern for extending graph builder,
# register a generic operator which only take care of builder generic ops in graph
# TODO: need to figure out a way to move this submodule to `utensor_cgen.ir` since graph
# builder is supposed to be generic but know it looks like utensor-specific
@OperatorFactory.register_generic_builder
@_export
class _GenericAddOperator(_Operator):
  op_type = 'AddOperator'

  @classmethod
  def build_op_info(cls, ugraph, name, tensor_x, tensor_y):
    # broadcast the shape and promote types
    dummy_x = np.empty(tensor_x.shape)
    dummy_y = np.empty(tensor_y.shape)
    output_shape = np.broadcast(dummy_x, dummy_y).shape
    output_dtype = np.promote_types(tensor_x.dtype, tensor_y.dtype)
    return OperationInfo(
      name=name,
      input_tensors=[tensor_x, tensor_y],
      output_tensors=[
      TensorInfo(
        name='{}:0'.format(name),
        op_name=name,
        dtype=output_dtype,
        shape=list(output_shape),
        ugraph=ugraph
      )
      ],
      op_type=cls.op_type,
      op_attr={
      'T': AttrValueConverter.__utensor_generic_type__(
        value_name='type',
        value=DataTypeConverter.get_tf_value(output_dtype)
      )
      },
      ugraph=ugraph,
      lib_name=ugraph.lib_name,
    )


@OperatorFactory.register_generic_builder
@_export
class _GenericConv2dOperator(_Operator):
  op_type = 'Conv2dOperator'

  @classmethod
  def build_op_info(cls, ugraph, name, tensor_x, tensor_w, stride_height, stride_width, padding='SAME'):
    # dboy: I'm too lazy to implement the padding algorithm again
    # simply call tf to find out the output shape
    dummy_x = np.empty(tensor_x.shape, dtype=tensor_x.dtype)
    dummy_w = np.empty(tensor_w.shape, dtype=tensor_w.dtype)
    graph = tf.Graph()
    with graph.as_default():
      dummy_out = tf.nn.conv2d(
        dummy_x,
        dummy_w,
        strides=[1, stride_height, stride_width, 1],
        padding=padding,
        name='dummy'
      )
    node_def = [node for node in graph.as_graph_def().node if node.name == 'dummy'][0]
    output_shape = dummy_out.shape.as_list()
    output_dtype = np.promote_types(tensor_x.dtype, tensor_w.dtype)
    op_attr = {
      k: AttrValueConverter.get_generic_value(v)
      for k, v in node_def.attr.items()
    }
    return OperationInfo(
      name=name,
      input_tensors=[tensor_x, tensor_w],
      output_tensors=[
        TensorInfo(
          name='{}:0'.format(name),
          op_name=name,
          dtype=output_dtype,
          shape=output_shape,
          ugraph=ugraph,
        )
      ],
      op_type=cls.op_type,
      op_attr=op_attr,
      ugraph=ugraph,
      lib_name=ugraph.lib_name,
    )


@OperatorFactory.register_generic_builder
@_export
class _GenericFullyConnectedOperator(_Operator):
  op_type = "FullyConnectedOperator"

  @classmethod
  def build_op_info(cls, ugraph, name, tensor_x, tensor_w, **kwargs):
    dtype_x = tensor_x.dtype
    dtype_w = tensor_w.dtype
    out_dtype = np.promote_types(dtype_x, dtype_w)
    if tensor_x.shape[-1] != tensor_w.shape[0]:
      raise ValueError(
        'dimension mismatch: {},{}'.format(tensor_x.shape, tensor_w.shape)
      )
    return OperationInfo(
      name=name,
      input_tensors=[
        tensor_x, tensor_w
      ],
      output_tensors=[
        TensorInfo(
          name='{}:0'.format(name),
          op_name=name,
          dtype=out_dtype,
          shape=tensor_x.shape[:-1]+tensor_w.shape[1:],
          ugraph=ugraph
        )
      ],
      op_type=cls.op_type,
      op_attr={
        # 'T': AttrValueConverter.__utensor_generic_type__(
        #   value_name='type',
        #   value=DataTypeConverter.get_tf_value(out_dtype)
        # ),
        # 'transpose_a': AttrValueConverter.__utensor_generic_type__(
        #   value_name='b',
        #   value=kwargs.get('transpose_x', False)
        # ),
        # 'transpose_b': AttrValueConverter.__utensor_generic_type__(
        #   value_name='b',
        #   value=kwargs.get('tranpose_w', False)
        # ),
        'FusedActivationFunction': '0 (NONE)',
      },
      ugraph=ugraph,
      lib_name=ugraph.lib_name,
    )


@OperatorFactory.register_generic_builder
@_export
class _GenericMaxOperator(_Operator):
  op_type = 'MaxOperator'

  @classmethod
  def build_op_info(cls, ugraph, name, tensor, axis=-1, keepdims=False):
    if isinstance(axis, int):
      axis, = ugraph.add_op(
        'Constant',
        values=np.array(axis, dtype=np.dtype('int32')),
        name='{}/axis'.format(name)
      )
    dummy_in = np.empty(tensor.shape, dtype=tensor.dtype)
    graph = tf.Graph()
    with graph.as_default():
      dummy_out = tf.reduce_max(
        dummy_in,
        axis=axis.op.op_attr['value'].value.np_array,
        keepdims=keepdims,
        name='dummy'
      )
    node_def = [node for node in graph.as_graph_def().node if node.name == 'dummy'][0]
    return OperationInfo(
      name=name,
      input_tensors=[tensor, axis],
      output_tensors=[
        TensorInfo(
          name='{}:0'.format(name),
          op_name=name,
          dtype=tensor.dtype,
          shape=dummy_out.shape.as_list(),
          ugraph=ugraph
        )
      ],
      op_type=cls.op_type,
      op_attr={
        k: AttrValueConverter.get_generic_value(v)
        for k, v in node_def.attr.items()
      },
      lib_name=ugraph.lib_name,
      ugraph=ugraph
    )


@OperatorFactory.register_generic_builder
@_export
class _GenericMaxPoolOperator(_Operator):
  op_type = 'MaxPoolOperator'

  @classmethod
  def build_op_info(
    cls,
    ugraph,
    name,
    tensor,
    ksize_height,
    ksize_width,
    stride_height,
    stride_width,
    padding='SAME',
  ):
    dummy_arr = np.empty(tensor.shape, dtype=tensor.dtype)
    graph = tf.Graph()
    with graph.as_default():
      tf_tensor = tf.nn.max_pool(
        dummy_arr,
        ksize=[1, ksize_height, ksize_width, 1],
        strides=[1, stride_height, stride_width, 1],
        padding=padding,
        name='dummy'
      )
    output_shape = tf_tensor.shape.as_list()
    graph_def = graph.as_graph_def()
    node_def = [node for node in graph_def.node if node.name == 'dummy'][0]
    return OperationInfo(
      name=name,
      input_tensors=[tensor],
      output_tensors=[
        TensorInfo(
          name='{}:0'.format(name),
          op_name=name,
          dtype=tensor.dtype,
          shape=output_shape,
          ugraph=ugraph
        )
      ],
      op_type=cls.op_type,
      lib_name=ugraph.lib_name,
      ugraph=ugraph,
      op_attr={
        k: AttrValueConverter.get_generic_value(v)
        for k, v in node_def.attr.items()
      }
    )


@OperatorFactory.register_generic_builder
@_export
class _GenericMinOperator(_Operator):
  op_type = 'MinOperator'

  @classmethod
  def build_op_info(cls, ugraph, name, tensor, axis=-1, keepdims=False):
    if isinstance(axis, int):
      axis, = ugraph.add_op(
        'Constant',
        values=np.array(axis, dtype=np.dtype('int32')),
        name='{}/axis'.format(name)
      )
    dummy_in = np.empty(tensor.shape, dtype=tensor.dtype)
    graph = tf.Graph()
    with graph.as_default():
      dummy_out = tf.reduce_min(
        dummy_in,
        axis=axis.op.op_attr['value'].value.np_array,
        keepdims=keepdims,
        name='dummy'
      )
    node_def = [node for node in graph.as_graph_def().node if node.name == 'dummy'][0]
    output_shape = dummy_out.shape.as_list()
    return OperationInfo(
      name=name,
      input_tensors=[tensor, axis],
      output_tensors=[
        TensorInfo(
          name='{}:0'.format(name),
          op_name=name,
          dtype=tensor.dtype,
          shape=output_shape,
          ugraph=ugraph,
        )
      ],
      op_type=cls.op_type,
      lib_name=ugraph.lib_name,
      ugraph=ugraph,
      op_attr={
        k: AttrValueConverter.get_generic_value(v)
        for k, v in node_def.attr.items()
      }
    )


@OperatorFactory.register_generic_builder
@_export
class _GenericReLUOperator(_Operator):
  op_type = "ReLUOperator"

  @classmethod
  def build_op_info(cls, ugraph, name, tensor, **kwargs):
    return OperationInfo(
      name=name,
      input_tensors=[tensor],
      output_tensors=[
        TensorInfo(
          name='{}:0'.format(name),
          op_name=name,
          dtype=tensor.dtype,
          shape=tensor.shape[:],
          ugraph=ugraph
        )
      ],
      op_type=cls.op_type,
      op_attr={
        'T': AttrValueConverter.__utensor_generic_type__(
          value_name='type',
          value=DataTypeConverter.get_tf_value(tensor.dtype)
        )
      },
      ugraph=ugraph,
      lib_name=ugraph.lib_name,
    )
