# -*- coding:utf8 -*-
import tensorflow as tf
from ._snippets_base import Snippet, register_template  # pylint: disable=W0611
from ._types import TYPES_MAP

__all__ = ["CreateTensorIdxSnippet", "CreateTensorNewSnippet",
           "AddOpSnippet", "MinOpSnippet", "MaxOpSnippet",
           "ArgMaxOpSnippet", "DequantizeOpSnippet",
           "MatMulOpSnippet"]


class CreateTensorIdxSnippet(Snippet):

  def __init__(self, data_dir, tensor_name,
               init_count=0,
               idx_fname=None,
               tf_dtype=tf.float32,
               sptr_name=None,
               create_sptr=False):
    if create_sptr and sptr_name is None:
      raise ValueError("sptr_name can't be None if create_sptr is True")
    if tf_dtype not in TYPES_MAP:
      raise ValueError("unsupport data type in uTensor: {}".format(tf_dtype))
    if idx_fname is None:
      idx_fname = "{}.idx".format(tensor_name.replace(":", "_").replace("/", "_"))
    Snippet.__init__(self, "create_tensor_idx.cpp")
    self.template_vars["data_dir"] = data_dir
    self.template_vars["tensor_name"] = tensor_name
    self.template_vars["init_count"] = init_count
    self.template_vars["idx_fname"] = idx_fname
    self.template_vars["dtype"] = TYPES_MAP[tf_dtype]
    self.template_vars["sptr_name"] = sptr_name
    self.template_vars["create_sptr"] = create_sptr


class CreateTensorNewSnippet(Snippet):

  def __init__(self, tensor_name,
               tensor_shape=None,
               init_count=0,
               idx_fname=None,
               tf_dtype=tf.float32,
               sptr_name=None,
               create_sptr=False):
    if create_sptr and sptr_name is None:
      raise ValueError("sptr_name can't be None if create_sptr is True")
    if tf_dtype not in TYPES_MAP:
      raise ValueError("unsupport data type in uTensor: {}".format(tf_dtype))
    if idx_fname is None:
      idx_fname = "{}.idx".format(tensor_name.replace(":", "_").replace("/", "_"))
    Snippet.__init__(self, "create_tensor_new.cpp")
    self.template_vars["tensor_name"] = tensor_name
    self.template_vars["tensor_shape"] = tensor_shape
    self.template_vars["init_count"] = init_count
    self.template_vars["idx_fname"] = idx_fname
    self.template_vars["dtype"] = TYPES_MAP[tf_dtype]
    self.template_vars["sptr_name"] = sptr_name
    self.template_vars["create_sptr"] = create_sptr


def _prepare_inputs(inputs):
  input_tnames = "{{{}}}".format(",".join(["\"{}\"".format(in_tensor) for in_tensor in inputs]))
  return input_tnames


class AddOpSnippet(Snippet):
  def __init__(self, inputs, output, tf_dtype=tf.float32):
    Snippet.__init__(self, "add_op.cpp")
    input_tnames = _prepare_inputs(inputs)
    output_tname = '{{"{}"}}'.format(output)
    self.template_vars["dtype"] = TYPES_MAP[tf_dtype]
    self.template_vars["input_tnames"] = input_tnames
    self.template_vars["output_tname"] = output_tname


class MinOpSnippet(Snippet):
  def __init__(self, inputs, output):
    Snippet.__init__(self, "min_op.cpp")
    input_tnames = _prepare_inputs(inputs)
    output_tname = '{{"{}"}}'.format(output)
    self.template_vars["input_tnames"] = input_tnames
    self.template_vars["output_tname"] = output_tname


class MaxOpSnippet(Snippet):
  def __init__(self, inputs, output):
    Snippet.__init__(self, "max_op.cpp")
    input_tnames = _prepare_inputs(inputs)
    output_tname = '{{"{}"}}'.format(output)
    self.template_vars["input_tnames"] = input_tnames
    self.template_vars["output_tname"] = output_tname


class ArgMaxOpSnippet(Snippet):
  def __init__(self, inputs, output, in_dtype=tf.float32, out_dtype=tf.int32):
    Snippet.__init__(self, "argmax_op.cpp")
    input_tnames = _prepare_inputs(inputs)
    output_tname = '{{"{}"}}'.format(output)
    self.template_vars["input_tnames"] = input_tnames
    self.template_vars["output_tname"] = output_tname
    self.template_vars["in_dtype"] = TYPES_MAP[in_dtype]
    self.template_vars["out_dtype"] = TYPES_MAP[out_dtype]


class DequantizeOpSnippet(Snippet):
  def __init__(self, inputs, output):
    Snippet.__init__(self, "dequantize_op.cpp")
    input_tnames = _prepare_inputs(inputs)
    output_tname = '{{"{}"}}'.format(output)
    self.template_vars["input_tnames"] = input_tnames
    self.template_vars["output_tname"] = output_tname


class MatMulOpSnippet(Snippet):
  def __init__(self):
    pass
