import re
from collections import defaultdict
from copy import deepcopy

import tensorflow as tf
import numpy as np

from utensor_cgen.ir import OperationInfo, uTensorGraph, TensorInfo
from utensor_cgen.ir.converter import AttrValueConverter, GenericTensorConverterMixin
from utensor_cgen.utils import parse_tensor_name
from utensor_cgen.ir.utils import graph_check

from utensor_cgen.experimental.ugraph_util_functions import *


__all__ = ["transpose_offline", "Const_Op", "Ram_Op", "Const_Reshape", "Uint8Q7Origin_Op",
           "CMSIS_FC_Op", "QuantRangeForMultiplicationu8u8int32_Op", "conv2d_op", "quantized_conv2d_op",
           "relu_op", "maxpool_op", "requantize_op", "requantization_range_op", "quantized_maxpool_op",
           "quantized_conv2d_pool_op"]

# Let us get unique names for custom injected nodes
def static_vars(**kwargs):
  def decorate(func):
    for k in kwargs:
      setattr(func, k, kwargs[k])
    return func
  return decorate

@static_vars(counters=defaultdict(int))
def get_unique_number(target):
  get_unique_number.counters[target] += 1
  return get_unique_number.counters[target]

def bs_ops_attr(np_array):
  """ Simplify creating the op_attr bullshit boilerplate from an numpy array """
  op_attr = {}
  op_attr["tensorflow_device"] = ''
  op_attr["dtype"] = AttrValueConverter.GenericType(value_name='type', value=3) # The hell are these values?
  op_attr["value"] = AttrValueConverter.GenericType(value_name='tensor', 
                                          value=GenericTensorConverterMixin.GenericType(np_array=np_array, 
                                                                                        dtype=np_array.dtype
                                                                                        )
                                         )
  return op_attr

def transpose_offline(op_info):
  #FIXME: might need to use deep copy on this one
  out_tensor_info = op_info.output_tensors[0]
  out_tensor_info.shape.reverse()
  transposed_value = op_info.op_attr['value'].value.np_array.transpose()
  op_info.op_attr['value'].value.np_array = transposed_value
  op_info.output_tensors[0] = out_tensor_info

  return op_info


def Const_Op(name, np_array, ugraph):
  tmp_graph = uTensorGraph(output_nodes=[name])
  const_tensor = TensorInfo(name=name + ":0",
                          op_name=name,
                          dtype=np_array.dtype,
                          shape=list(np_array.shape),
                          ugraph=tmp_graph
                          )
  const_op_info = OperationInfo(name=name,
                          input_tensors=list(),
                          output_tensors=[const_tensor],
                          op_type="Const",
                          backend="tensorflow",
                          ugraph=tmp_graph,
                          op_attr=bs_ops_attr(np_array)
                          )
  ugraph.add_op(const_op_info, False)

  return const_op_info.output_tensors

def Ram_Op(name, np_array, ugraph):
  out = Const_Op(name, np_array, ugraph)
  ugraph.ops_info[name].op_type = "Ram"

  return out

def Reshape_Op(name, input_tensor, shape_tensor, ugraph):
  #FIXME: ValueError: duplicate op detected
  #if ugraph == None:
  tmp_ugraph = uTensorGraph(output_nodes=[name])

  shape_const_op = ugraph.ops_info[shape_tensor[0].op_name]

  reshape_out_tensor = TensorInfo(name=name + ":0",
                            op_name=name,
                            dtype=input_tensor[0].dtype,
                            shape=shape_const_op.op_attr['value'].value.np_array.tolist(),
                            ugraph=tmp_ugraph
                            )

  reshape_opInfo = OperationInfo(name=name,
                          input_tensors=[input_tensor[0], shape_tensor[0]],
                          output_tensors=[reshape_out_tensor],
                          op_type="Reshape",
                          backend="tensorflow",
                          ugraph=tmp_ugraph
                          )

  ugraph.add_op(reshape_opInfo, False)

  return reshape_opInfo.output_tensors

def Const_Reshape(name, input_tensor, shape, ugraph):
  const_name = name + "_const"

  for i, v in enumerate(shape):
    if v == None:
      shape[i] = 1

  reshape_const_tensor = Const_Op(const_name, np.array(shape), ugraph)
  return Reshape_Op(name, input_tensor, reshape_const_tensor, ugraph)

def Uint8Q7Origin_Op(name, inputs, ugraph):
  tmp_ugraph = uTensorGraph(output_nodes=[name])
  q7_out = TensorInfo(name=name + ":0",
                    op_name=name,
                    dtype=np.dtype('int8'),
                    shape=inputs[0].shape,
                    ugraph=tmp_ugraph
                    )
  q7_op_info = OperationInfo(name=name,
                        input_tensors=inputs,
                        output_tensors=[q7_out],
                        op_type="Uint8Q7OriginOp",
                        backend="tensorflow",
                        ugraph=tmp_ugraph)
  
  # if(name == 'convert_uint8_q7_Relu/eightbit_transpose_0_q7'):
  # import pdb; pdb.set_trace()
  ugraph.add_op(q7_op_info, False)

  return q7_op_info.output_tensors

def CMSIS_FC_Op(name, in0, in1, bias, bShift, oShift, scratch, ugraph):
  tmp_ugraph = uTensorGraph(output_nodes=[name])

  #in reality, CMSIS_FC is doing in1 * in0
  out_shape = [in1[0].shape[0], in0[0].shape[1]]
  # for i, v in enumerate(out_shape):
  #   if v == None:
  #     out_shape[i] = 1
  #     break

  fc_out_tensor = TensorInfo(name=name + ":0",
                    op_name=name,
                    dtype=np.dtype('int32'), #hard coding to int32
                    shape=out_shape,
                    ugraph=tmp_ugraph
                    )

  fc_op_info = OperationInfo(name=name,
                          input_tensors=[in0[0], in1[0], bias[0], bShift[0], oShift[0], scratch[0]],
                          output_tensors=[fc_out_tensor],
                          op_type="CMSIS_NN_FC",
                          backend="tensorflow",
                          ugraph=tmp_ugraph
                          )
                          
  ugraph.add_op(fc_op_info, False)
  return fc_op_info.output_tensors

def QuantRangeForMultiplicationu8u8int32_Op(name, a_range, b_range, ugraph):

  tmp_ugraph = uTensorGraph(output_nodes=[name])
  min_out = TensorInfo(name=name + ":0",
                    op_name=name,
                    dtype=np.dtype('float'),
                    shape=[1],
                    ugraph=tmp_ugraph
                    )
  max_out = TensorInfo(name=name + ":1",
                  op_name=name,
                  dtype=np.dtype('float'),
                  shape=[1],
                  ugraph=tmp_ugraph
                  )

  new_range_op_info = OperationInfo(name=name,
                    input_tensors=[a_range[0], a_range[1], b_range[0], b_range[1]],
                    output_tensors=[min_out, max_out],
                    op_type="QuantRangeForMultiplicationu8u8int32Op",
                    backend="tensorflow",
                    ugraph=tmp_ugraph
                    )

  ugraph.add_op(new_range_op_info, False)

  return new_range_op_info.output_tensors

def conv2d_op(name, inputs, ugraph):
  tmp_ugraph = uTensorGraph(output_nodes=[name])
  conv_out = TensorInfo(name=name + ":0",
                    op_name=name,
                    dtype=np.dtype('float32'),
                    shape=inputs[0].shape,  #FIXME: wrong shape most likely
                    ugraph=tmp_ugraph
                    )
  conv2d_op_info = OperationInfo(name=name,
                        input_tensors=inputs,
                        output_tensors=[conv_out],
                        op_type="Conv2D",
                        backend="tensorflow",
                        ugraph=tmp_ugraph)
  
  ugraph.add_op(conv2d_op_info, False)
  return conv2d_op_info.output_tensors

def quantized_conv2d_op(name, inputs, ugraph):
  tmp_ugraph = uTensorGraph(output_nodes=[name])
  conv_out = TensorInfo(name=name + ":0",
                    op_name=name,
                    dtype=np.dtype('uint8'),
                    shape=inputs[0].shape, #FIXME: wrong shape most likely
                    ugraph=tmp_ugraph
                    )
  min_out = TensorInfo(name=name + ":1",
            op_name=name,
            dtype=np.dtype('float32'),
            shape=[1],
            ugraph=tmp_ugraph
            )
  max_out = TensorInfo(name=name + ":2",
              op_name=name,
              dtype=np.dtype('float32'),
              shape=[1],
              ugraph=tmp_ugraph
              )

  quantized_conv2d_op_info = OperationInfo(name=name,
                        input_tensors=inputs,
                        output_tensors=[conv_out, min_out, max_out],
                        op_type="QuantizedConv2D",
                        backend="tensorflow",
                        ugraph=tmp_ugraph)

  ugraph.add_op(quantized_conv2d_op_info, False)
  return quantized_conv2d_op_info.output_tensors

def relu_op(name, inputs, ugraph):
  tmp_ugraph = uTensorGraph(output_nodes=[name])
  relu_out = TensorInfo(name=name + ":0",
                    op_name=name,
                    dtype=np.dtype('float32'),
                    shape=inputs[0].shape,
                    ugraph=tmp_ugraph
                    )
  relu_op_info = OperationInfo(name=name,
                        input_tensors=inputs,
                        output_tensors=[relu_out],
                        op_type="Relu",
                        backend="tensorflow",
                        ugraph=tmp_ugraph)
  
  ugraph.add_op(relu_op_info, False)
  return relu_op_info.output_tensors

def maxpool_op(name, inputs, ugraph):
  tmp_ugraph = uTensorGraph(output_nodes=[name])
  max_out = TensorInfo(name=name + ":0",
                    op_name=name,
                    dtype=np.dtype('float32'),
                    shape=inputs[0].shape,
                    ugraph=tmp_ugraph
                    )
  max_op_info = OperationInfo(name=name,
                        input_tensors=inputs,
                        output_tensors=[max_out],
                        op_type="MaxPool",
                        backend="tensorflow",
                        ugraph=tmp_ugraph)
  ugraph.add_op(max_op_info, False)
  return max_op_info.output_tensors

def quantized_maxpool_op(name, inputs, ugraph):
  tmp_ugraph = uTensorGraph(output_nodes=[name])
  max_value_out = TensorInfo(name=name + ":0",
                    op_name=name,
                    dtype=inputs[0].dtype,
                    shape=inputs[0].shape,
                    ugraph=tmp_ugraph
                    )
  min_out = TensorInfo(name=name + ":1",
            op_name=name,
            dtype=np.dtype('float32'),
            shape=[1],
            ugraph=tmp_ugraph
            )
  max_out = TensorInfo(name=name + ":2",
              op_name=name,
              dtype=np.dtype('float32'),
              shape=[1],
              ugraph=tmp_ugraph
              )

  qnt_max_op_info = OperationInfo(name=name,
                        input_tensors=inputs,
                        output_tensors=[max_value_out, min_out, max_out],
                        op_type="QuantizedMaxPool",
                        backend="tensorflow",
                        ugraph=tmp_ugraph)
  ugraph.add_op(qnt_max_op_info, False)
  return qnt_max_op_info.output_tensors

def requantization_range_op(name, inputs, ugraph):
  tmp_ugraph = uTensorGraph(output_nodes=[name])

  min_out = TensorInfo(name=name + ":0",
              op_name=name,
              dtype=np.dtype('float32'),
              shape=[1],
              ugraph=tmp_ugraph
              )
  max_out = TensorInfo(name=name + ":1",
              op_name=name,
              dtype=np.dtype('float32'),
              shape=[1],
              ugraph=tmp_ugraph
              )

  rqntr_op_info = OperationInfo(name=name,
                    input_tensors=inputs,
                    output_tensors=[min_out, max_out],
                    op_type="RequantizationRange",
                    backend="tensorflow",
                    ugraph=tmp_ugraph)

  ugraph.add_op(rqntr_op_info, False)
  return rqntr_op_info.output_tensors


def requantize_op(name, inputs, ugraph):
  tmp_ugraph = uTensorGraph(output_nodes=[name])

  value_out = TensorInfo(name=name + ":0",
              op_name=name,
              dtype=np.dtype('uint8'),
              shape=inputs[0].shape,
              ugraph=tmp_ugraph
              )

  min_out = TensorInfo(name=name + ":1",
              op_name=name,
              dtype=np.dtype('float32'),
              shape=[1],
              ugraph=tmp_ugraph
              )
  max_out = TensorInfo(name=name + ":2",
              op_name=name,
              dtype=np.dtype('float32'),
              shape=[1],
              ugraph=tmp_ugraph
              )

  rqnt_op_info = OperationInfo(name=name,
                    input_tensors=inputs,
                    output_tensors=[value_out, min_out, max_out],
                    op_type="Requantize",
                    backend="tensorflow",
                    ugraph=tmp_ugraph)

  ugraph.add_op(rqnt_op_info, False)
  return [value_out, min_out, max_out]


#FIXME: The content in op_attr should be passed in as proper parameters
def quantized_conv2d_pool_op(name, inputs, op_attr, ugraph):
  tmp_ugraph = uTensorGraph(output_nodes=[name])
  conv_out = TensorInfo(name=name + ":0",
                    op_name=name,
                    dtype=inputs[0].dtype,
                    shape=inputs[0].shape, #FIXME: wrong shape most likely
                    ugraph=tmp_ugraph
                    )
  min_out = TensorInfo(name=name + ":1",
            op_name=name,
            dtype=np.dtype('float32'),
            shape=[1],
            ugraph=tmp_ugraph
            )
  max_out = TensorInfo(name=name + ":2",
              op_name=name,
              dtype=np.dtype('float32'),
              shape=[1],
              ugraph=tmp_ugraph
              )

  quantized_conv2d_op_info = OperationInfo(name=name,
                        input_tensors=inputs,
                        output_tensors=[conv_out, min_out, max_out],
                        op_type="QuantizedFusedConv2DMaxpool",
                        backend="tensorflow",
                        op_attr=op_attr,
                        ugraph=tmp_ugraph)

  ugraph.add_op(quantized_conv2d_op_info, False)
  return quantized_conv2d_op_info.output_tensors
