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


__all__ = ["transpose_offline", "create_const_op", "create_ram_op", "create_const_reshape"]

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
  out_tensor_info = op_info.output_tensors[0]
  out_tensor_info.shape.reverse()
  transposed_value = op_info.op_attr['value'].value.np_array.transpose()
  op_info.op_attr['value'].value.np_array = transposed_value
  op_info.output_tensors[0] = out_tensor_info

  return op_info


def create_const_op(name, np_array, ugraph=None):
  if ugraph == None:
    ugraph = uTensorGraph()
  const_tensor = TensorInfo(name=name + ":0",
                          op_name=name,
                          dtype=np_array.dtype,
                          shape=list(np_array.shape),
                          ugraph=ugraph
                          )
  const_op_info = OperationInfo(name=name,
                          input_tensors=list(),
                          output_tensors=[const_tensor],
                          op_type="Const",
                          backend="tensorflow",
                          ugraph=ugraph,
                          op_attr=bs_ops_attr(np_array)
                          )
  output = list()
  output.append(const_tensor)

  return (const_op_info, output)

def create_ram_op(name, np_array, ugraph=None):
  (op_info, outs) = create_const_op(name, np_array, ugraph)
  op_info.op_type = "Ram"
  return (op_info, outs)

def create_reshape_op(name, inputs, ugraph):
  #FIXME: ValueError: duplicate op detected
  #if ugraph == None:
  tmp_ugraph = uTensorGraph()

  shape_const_op = ugraph.ops_info[inputs[1].op_name]

  reshape_out_tensor = TensorInfo(name=name + ":0",
                            op_name=name,
                            dtype=inputs[0].dtype,
                            shape=shape_const_op.op_attr['value'].value.np_array.tolist(),
                            ugraph=tmp_ugraph
                            )

  reshape_opInfo = OperationInfo(name=name,
                          input_tensors=inputs,
                          output_tensors=[reshape_out_tensor],
                          op_type="Reshape",
                          backend="tensorflow",
                          ugraph=tmp_ugraph
                          )
  return (reshape_opInfo, [reshape_out_tensor])

def create_const_reshape(name, inputs, shape, ugraph):
  const_name = name + "_const"
  (reshape_const_opInfo, reshape_const_tensors) = create_const_op(const_name, np.array(shape))
  ugraph.add_op(reshape_const_opInfo)
  return create_reshape_op(name, [inputs[0], reshape_const_tensors[0]], ugraph)