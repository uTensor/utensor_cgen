# -*- coding:utf8 -*-
import numpy as np
import tensorflow as tf

from ._snippets_base import Snippet, SnippetContainer  # pylint: disable=W0611
from ._types import TF_TYPES_MAP

__all__ = ["CreateTensorIdxSnippet", "CreateTensorNewSnippet",
           "AddOpSnippet", "MinOpSnippet", "MaxOpSnippet",
           "ArgMaxOpSnippet", "DequantizeOpSnippet",
           "QuantizedMatMulOpSnippet", "QuantizeV2OpSnippet",
           "QuantizedReluOpSnippet", "ReshapeOpSnippet",
           "RequantizationRangeOpSnippet", "RequantizeOpSnippet",
           "CommentSnippet", "ContextHeaderSnippet"]


class CreateTensorIdxSnippet(Snippet):
  __template_name__ = "snippets/create_tensor_idx.cpp"

  def __init__(self, data_dir, tensor_name,
               ref_count=0,
               idx_fname=None,
               tf_dtype=tf.float32,
               sptr_name=None,
               create_sptr=False):
    if create_sptr and sptr_name is None:
      raise ValueError("sptr_name can't be None if create_sptr is True")
    if tf_dtype not in TF_TYPES_MAP:
      raise ValueError("unsupport data type in uTensor: {}".format(tf_dtype))
    if idx_fname is None:
      idx_fname = "{}.idx".format(tensor_name.replace(":", "_").replace("/", "_"))
    Snippet.__init__(self)
    self.template_vars["data_dir"] = data_dir
    self.template_vars["tensor_name"] = tensor_name
    self.template_vars["ref_count"] = ref_count
    self.template_vars["idx_fname"] = idx_fname
    self.template_vars["importer_dtype"] = TF_TYPES_MAP[tf_dtype].importer_type_str
    self.template_vars["sptr_name"] = sptr_name
    self.template_vars["create_sptr"] = create_sptr


class CreateTensorNewSnippet(Snippet):
  __template_name__ = "snippets/create_tensor_new.cpp"

  def __init__(self, tensor_name,
               tensor_shape=None,
               ref_count=0,
               idx_fname=None,
               tf_dtype=tf.float32,
               sptr_name=None,
               create_sptr=False):
    if create_sptr and sptr_name is None:
      raise ValueError("sptr_name can't be None if create_sptr is True")
    if tf_dtype not in TF_TYPES_MAP:
      raise ValueError("unsupport data type in uTensor: {}".format(tf_dtype))
    if idx_fname is None:
      idx_fname = "{}.idx".format(tensor_name.replace(":", "_").replace("/", "_"))

    Snippet.__init__(self)
    self.template_vars["tensor_name"] = tensor_name
    self.template_vars["tensor_shape"] = tensor_shape
    self.template_vars["ref_count"] = ref_count
    self.template_vars["idx_fname"] = idx_fname
    self.template_vars["dtype"] = TF_TYPES_MAP[tf_dtype].tensor_type_str
    self.template_vars["sptr_name"] = sptr_name
    self.template_vars["create_sptr"] = create_sptr

  def _to_shape_str(self, shape):
    shape_str = ",".join([str(dim) for dim in shape])
    return "{" + shape_str + "}"


# TODO fix uTensor..?
def _prepare_inputs(inputs):
  input_tnames = "{{{}}}".format(",".join(["\"{}\"".format(in_tensor) for in_tensor in inputs]))
  return input_tnames


def _permute_args(args, perm=None):
  if perm is None:
    perm = [i for i in range(len(args))]
  return [arg for arg in np.array(args)[perm]]


class AddOpSnippet(Snippet):
  __template_name__ = "snippets/add_op.cpp"

  def __init__(self, inputs, output, tf_dtype=tf.float32, ref_count=0):
    Snippet.__init__(self)
    self.template_vars["in_dtype"] = TF_TYPES_MAP[tf_dtype].tensor_type_str
    self.template_vars["out_dtype"] = TF_TYPES_MAP[tf_dtype].tensor_type_str
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["ref_count"] = ref_count


class MinOpSnippet(Snippet):
  __template_name__ = "snippets/min_op.cpp"

  def __init__(self, inputs, output, out_dtype, out_shape=None, ref_count=0):
    Snippet.__init__(self)
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["out_dtype"] = TF_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["out_shape"] = out_shape
    self.template_vars["ref_count"] = ref_count


class MaxOpSnippet(Snippet):
  __template_name__ = "snippets/max_op.cpp"

  def __init__(self, inputs, output, out_dtype, out_shape=None, ref_count=0):
    Snippet.__init__(self)
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["out_dtype"] = TF_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["out_shape"] = out_shape
    self.template_vars["ref_count"] = ref_count


class ArgMaxOpSnippet(Snippet):
  __template_name__ = "snippets/argmax_op.cpp"

  def __init__(self, inputs, output, in_dtype=tf.float32, out_dtype=tf.int32, ref_count=0):
    Snippet.__init__(self)
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["in_dtype"] = TF_TYPES_MAP[in_dtype].tensor_type_str
    self.template_vars["out_dtype"] = TF_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["ref_count"] = ref_count


class DequantizeOpSnippet(Snippet):
  __template_name__ = "snippets/dequantize_op.cpp"

  def __init__(self, inputs, output, out_dtype, ref_count=0):
    Snippet.__init__(self)
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["out_dtype"] = TF_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["ref_count"] = ref_count


class QuantizedMatMulOpSnippet(Snippet):
  __template_name__ = "snippets/qmatmul_op.cpp"

  def __init__(self, inputs, outputs, x_dtype, w_dtype, out_dtype, ref_counts=None):
    Snippet.__init__(self)
    # hack on different arguments order between tensorflow and uTensor
    inputs = _permute_args(inputs, [0, 2, 3, 1, 4, 5])
    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["x_dtype"] = TF_TYPES_MAP[x_dtype].tensor_type_str
    self.template_vars["w_dtype"] = TF_TYPES_MAP[w_dtype].tensor_type_str
    self.template_vars["out_dtype"] = TF_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars['ref_counts'] = ref_counts


class QuantizeV2OpSnippet(Snippet):
  __template_name__ = "snippets/quantV2_op.cpp"

  def __init__(self, inputs, outputs, out_dtype, ref_counts=None):
    Snippet.__init__(self)
    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["out_dtype"] = TF_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["ref_counts"] = ref_counts


class QuantizedReluOpSnippet(Snippet):
  __template_name__ = "snippets/qrelu_op.cpp"

  def __init__(self, inputs, outputs, in_dtype, out_dtypes, qout_dtype, ref_counts=None):
    Snippet.__init__(self)
    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["in_dtype"] = TF_TYPES_MAP[in_dtype].tensor_type_str
    self.template_vars["out_dtypes"] = [TF_TYPES_MAP[out_dtype].tensor_type_str for out_dtype in out_dtypes]
    self.template_vars["qout_dtype"] = TF_TYPES_MAP[qout_dtype].tensor_type_str
    self.template_vars["ref_counts"] = ref_counts


class RequantizationRangeOpSnippet(Snippet):
  __template_name__ = "snippets/requant_range_op.cpp"

  def __init__(self, inputs, outputs, out_dtype, ref_counts=None):
    Snippet.__init__(self)
    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["out_dtype"] = TF_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["ref_counts"] = ref_counts


class RequantizeOpSnippet(Snippet):
  __template_name__ = "snippets/requant_op.cpp"

  def __init__(self, inputs, outputs, qout_dtype, range_dtype, ref_counts=None):
    """qout_dtype: Tout
    range_dtype: T2
    input_dtype: T1
    """
    Snippet.__init__(self)
    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["qout_dtype"] = TF_TYPES_MAP[qout_dtype].tensor_type_str
    self.template_vars["range_dtype"] = TF_TYPES_MAP[range_dtype].tensor_type_str
    self.template_vars["ref_counts"] = ref_counts


class ReshapeOpSnippet(Snippet):
  __template_name__ = "snippets/reshape_op.cpp"

  def __init__(self, inputs, output, ref_count=0):
    Snippet.__init__(self)
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["ref_count"] = ref_count


class CommentSnippet(Snippet):
  __template_name__ = "snippets/comments.cpp"

  def __init__(self, comments):
    Snippet.__init__(self)
    self.template_vars["comments"] = comments


class ContextHeaderSnippet(Snippet):
  __template_name__ = "snippets/get_ctx.hpp"

  def __init__(self, guard_name, graph_name, placeholders=None):
    Snippet.__init__(self)
    if placeholders is None:
      placeholders = []
    self.template_vars["header_guard"] = "_{}_H".format(guard_name.upper())
    self.template_vars["graph_name"] = graph_name
    self.template_vars["placeholders"] = placeholders


class ContextSnippetsContainer(SnippetContainer):
  __template_name__ = "containers/get_ctx.cpp"

  def __init__(self, graph_name, ctx_header_name, snippets=None, placeholders=None, ref_counts=None):
    SnippetContainer.__init__(self, snippets)
    if placeholders is None:
      placeholders = []
    if ref_counts is None:
      ref_counts = []
    self.template_vars["graph_name"] = graph_name
    self.template_vars["placeholders"] = placeholders
    self.template_vars["ref_counts"] = ref_counts
    self.add_header('"{}"'.format(ctx_header_name))
