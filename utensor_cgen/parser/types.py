# -*- coding: utf8 -*-
import six
import attr
from attr import validators
import numpy as np
import tensorflow as tf


@attr.s
class TensorInfo(object):
  name = attr.ib(validator=validators.instance_of(six.text_type))
  dtype = attr.ib(validator=validators.instance_of(tf.DType))
  shape = attr.ib()

  @shape.validator
  def check(self, attribute, value):
    if value is not None and not isinstance(value, list):
      raise ValueError('shape must be None or list')

  # legacy code: to make it works like namedtuple
  def __iter__(self):
    # TODO remove all such code in utensor
    #   name, dtype, shape = tensor_info
    return iter((self.name, self.dtype, self.shape))

  def __getitem__(self, k):
    return (self.name, self.dtype, self.shape)[k]


@attr.s
class OperationInfo(object):
  node_name = attr.ib(type=six.text_type)

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

  op_type = attr.ib(validator=validators.instance_of(six.text_type))

  output_content = attr.ib(validator=validators.instance_of(dict))

  @output_content.validator
  def check(self, attribute, value):
    if not all([isinstance(k, six.text_type) for k in value.keys()]):
      raise ValueError('All key for output_content should be of type six.text_type')
    if not all([isinstance(v, np.ndarray) for v in value.values()]):
      raise ValueError('All value of output_content should be of type numpy.ndarray')

  op_attr = attr.ib(default=None)
