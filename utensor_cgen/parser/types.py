# -*- coding: utf8 -*-
import attr
from attr import validators
import numpy as np
import tensorflow as tf


@attr.s
class TensorInfo(object):
  name = attr.ib(validator=validators.instance_of(str))
  dtype = attr.ib(validator=validators.instance_of(type(tf.float32)))
  shape = attr.ib(validator=validators.instance_of(list))

  def __iter__(self):
    # legacy code
    # TODO remove all such code in utensor
    #   name, dtype, shape = tensor_info
    return iter((self.name, self.dtype, self.shape))


@attr.s
class OperationInfo(object):
  node_name = attr.ib(type=str)

  input_tensor = attr.ib(validator=validators.instance_of(list))

  @input_tensor.validator
  def check(self, attribute, value):
    if not all([isinstance(v, TensorInfo) for v in value]):
      raise ValueError('Expecting a list of TensorInfo for input_tensor')

  output_tensor = attr.ib(validator=validators.instance_of(list))

  @output_tensor.validator
  def check(self, attribute, value):
    if not all([isinstance(v, TensorInfo) for v in value]):
      raise ValueError('Expecting a list of TensorInfo for output_tensor')

  op_type = attr.ib(validator=validators.instance_of(str))

  output_content = attr.ib(validator=validators.instance_of(dict))

  @output_content.validator
  def check(self, attribute, value):
    if not all([isinstance(k, str) for k in value.keys()]):
      raise ValueError('All key for output_content should be of type str')
    if not all([isinstance(v, np.ndarray) for v in value.values()]):
      raise ValueError('All value of output_content should be of type numpy.ndarray')

  op_attr = attr.ib(default=None)
