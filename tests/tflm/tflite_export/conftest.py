import numpy as np
from pytest import fixture

import tensorflow as tf
from utensor_cgen.ir import TensorInfo, OperationInfo, uTensorGraph
from utensor_cgen.ir.converter import (AttrValueConverter, DataTypeConverter,
                                       GenericTensorConverterMixin)
from utensor_cgen.utils import prune_graph, topologic_order_graph
from utensor_cgen.backend.operators import OperatorFactory, _Operator
from utensor_cgen.matcher import OpEqualityDelegate, _morphism


@OperatorFactory.register
@OpEqualityDelegate.is_associative(
  permutations=((0, 1), (1, 0))
)
class _TFLM_AddOperator(_Operator):

  op_type = "TFLM_ADD" # tf op type

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    output = op_info.output_tensors[0].name
    tf_dtype = op_info.input_tensors[0].dtype

  @classmethod
  def build_op_info(cls, ugraph, name, tensor_x, tensor_y, **kwargs):
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
      backend=kwargs.get('backend', 'TFLM')
    )


@OperatorFactory.register
class _TFLM_FULLY_CONNECTED_Operator(_Operator):

  op_type="TFLM_FULLY_CONNECTED"
  
  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    output = op_info.output_tensors[0].name
    out_dtype = op_info.output_tensors[0].dtype
    in_dtypes = [tensor_info.dtype for tensor_info in op_info.input_tensors]
    #assert (op_info.input_tensors[0].shape[1] == None or op_info.input_tensors[0].shape[1] == 1)

  @classmethod
  def build_op_info(cls, ugraph, name, tensor_x, tensor_w, tensor_b, **kwargs):
    output_shape = [tensor_w.shape[0], tensor_x.shape[1]]
    #output_dtype = np.promote_types(tensor_x.dtype, tensor_y.dtype)
    output_dtype = tensor_x.dtype
    return OperationInfo(
      name=name,
      input_tensors=[tensor_x, tensor_w, tensor_b],
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
      backend=kwargs.get('backend', 'TFLM')
    )

@fixture(name='hybrid_quant_output')
def simple_tflm_graph():
    ugraph = uTensorGraph()

    with ugraph.begin_construction():
        tensor_x0, = ugraph.add_op(
            op_type='Const',
            name='x0',
            value=np.array([1, 1, 1, 1], dtype=np.float32)[:, np.newaxis]
        )
        tensor_x1, = ugraph.add_op(
            op_type='Const',
            name='x1',
            value=np.array([2, 4, 6, 8], dtype=np.float32)[:, np.newaxis]
        )
        tensor_w, = ugraph.add_op(
            op_type='Const',
            name='w',
            value=np.array([10, 20, 30, 40], dtype=np.float32)[np.newaxis, :]
        )
        tensor_b, = ugraph.add_op(
            op_type='Const',
            name='b',
            value=np.array([7], dtype=np.float32)
        )
        

        tensor_addout, = ugraph.add_op(
            tensor_x0, tensor_x1,
            op_type='TFLM_ADD',
            name='TFLM_ADD0'
        )

        tensor_out, = ugraph.add_op(
            tensor_addout, tensor_w, tensor_b,
            op_type='TFLM_FULLY_CONNECTED',
            name='TFLM_FULLY_CONNECTED00',
            is_output=True
        )

    return [ugraph, ["x0:0", "x1:0"], ["w:0", "b:0", tensor_out.name]]
