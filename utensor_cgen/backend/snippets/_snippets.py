# -*- coding:utf8 -*-
import numpy as np

from ._base import Snippet, SnippetContainerBase  # pylint: disable=W0611
from ._types import NP_TYPES_MAP

__all__ = ["Snippet", "SnippetContainerBase",
           "CreateTensorIdxSnippet", "CreateTensorNewSnippet",
           "AddOpSnippet", "MinOpSnippet", "MaxOpSnippet",
           "ArgMaxOpSnippet", "DequantizeOpSnippet",
           "QuantizedMaxPoolSnippet", "MaxPoolSnippet",
           "QuantizedMatMulOpSnippet", "MatMulOpSnippet", "QuantizeV2OpSnippet",
           "ReluOpSnippet", "QuantizedReluOpSnippet", "ShapeOpSnippet",
           "StridedSliceOpSnippet", "PackOpSnippet", "SoftmaxOpSnippet",
           "ReshapeOpSnippet", "QuantizedReshapeOpSnippet",
           "Conv2DOpSnippent", "Conv2DQuantOpSnippent", "CMSISNNFCOpSnippet",
           "RequantizationRangeOpSnippet", "RequantizeOpSnippet",
           "CommentSnippet", "ContextHeaderSnippet",
           "ContextSnippetsContainer", "QuantizedAddOpSnippet",
           "CreateTensorBinarySnippet", "WeightSnippet",
           "ContextGlobalArrayContainer", "QuantRangeForMultiplicationSnippet",
           "CreateTensorRamSnippet", "Uint8Q7OriginSnippet"]

# TODO: Better abstraction, i.e a better backend for code generation
class CreateTensorIdxSnippet(Snippet):
  __template_name__ = "snippets/create_tensor_idx.cpp"
  __headers__ = set(['"uTensor/loaders/tensorIdxImporter.hpp"',
                     '"uTensor/core/context.hpp"',
                     '"uTensor/core/tensor.hpp"'])

  def __init__(self, data_dir, tensor_name, np_dtype,
               ref_count=0,
               idx_fname=None,
               sptr_name=None,
               create_sptr=False,
               to_eval=False):
    if create_sptr and sptr_name is None:
      raise ValueError("sptr_name can't be None if create_sptr is True")
    if np_dtype not in NP_TYPES_MAP:
      raise ValueError("unsupport data type in uTensor: {}".format(np_dtype))
    if idx_fname is None:
      idx_fname = "{}.idx".format(tensor_name.replace(":", "_").replace("/", "_"))
    Snippet.__init__(self)
    idx_path = "{}/{}".format(data_dir, idx_fname)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    if create_sptr:
      self.template_vars["create_sptr"] = create_sptr
      self.template_vars["sptr_name"] = sptr_name
    self.template_vars["idx_path"] = idx_path
    self.template_vars["tensor_name"] = tensor_name
    self.template_vars["importer_dtype"] = NP_TYPES_MAP[np_dtype].importer_type_str
    self.template_vars["to_eval"] = to_eval

class CreateTensorRamSnippet(Snippet):
  __template_name__ = "snippets/create_tensor_new.cpp"
  __headers__ = set(['"uTensor/core/context.hpp"',
                     '"uTensor/core/tensor.hpp"'])

  def __init__(self, tensor_name, tf_dtype, tensor_shape=None,
               ref_count=0,
               sptr_name=None,
               create_sptr=False,
               to_eval=False):
    if create_sptr and sptr_name is None:
      raise ValueError("sptr_name can't be None if create_sptr is True")
    if tf_dtype not in NP_TYPES_MAP:
      raise ValueError("unsupport data type in uTensor: {}".format(tf_dtype))
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    if create_sptr:
      self.template_vars["create_sptr"] = create_sptr
      self.template_vars["sptr_name"] = sptr_name
    self.template_vars["tensor_type"] = "RamTensor"
    self.template_vars["tensor_name"] = tensor_name
    self.template_vars["tensor_shape"] = self._to_shape_str(tensor_shape)
    self.template_vars["dtype"] = NP_TYPES_MAP[tf_dtype].tensor_type_str
    self.template_vars["to_eval"] = to_eval

  def _to_shape_str(self, shape):
    shape_str = ",".join([str(dim) for dim in shape])
    return "{" + shape_str + "}"


class CreateTensorBinarySnippet(Snippet):
  __template_name__ = "snippets/create_tensor_binary.cpp"
  __headers__ = set(['"uTensor/core/context.hpp"',
                     '"uTensor/core/tensor.hpp"'])

  def __init__(self, tensor_name, tf_dtype, tensor_shape=None,
               ref_count=0,
               sptr_name=None,
               inline_name=None,
               create_sptr=False,
               to_eval=False):
    if create_sptr and sptr_name is None:
      raise ValueError("sptr_name can't be None if create_sptr is True")
    if tf_dtype not in NP_TYPES_MAP:
      raise ValueError("unsupport data type in uTensor: {}".format(tf_dtype))
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    if create_sptr:
      self.template_vars["create_sptr"] = create_sptr
      self.template_vars["sptr_name"] = sptr_name
    self.template_vars["tensor_type"] = "BinaryTensor"
    self.template_vars["tensor_name"] = tensor_name
    #FIXME: a patch to make scalar RomTensor compilable: [] vs [1]
    if tensor_shape == []:
      tensor_shape = [1]
    self.template_vars["tensor_shape"] = self._to_shape_str(tensor_shape)
    self.template_vars["tensor_length"] = np.prod(tensor_shape)
    self.template_vars["dtype"] = NP_TYPES_MAP[tf_dtype].tensor_type_str
    self.template_vars["to_eval"] = to_eval
    self.template_vars["inline_name"] = inline_name

  def _to_shape_str(self, shape):
    shape_str = ",".join([str(dim) for dim in shape])
    return "{" + shape_str + "}"


class CreateTensorNewSnippet(Snippet):
  __template_name__ = "snippets/create_tensor_new.cpp"
  __headers__ = set(['"uTensor/core/context.hpp"', '"uTensor/core/tensor.hpp"'])

  def __init__(self, tensor_name, np_dtype,
               tensor_shape=None,
               ref_count=0,
               idx_fname=None,
               sptr_name=None,
               create_sptr=False,
               to_eval=False):
    if create_sptr and sptr_name is None:
      raise ValueError("sptr_name can't be None if create_sptr is True")
    if np_dtype not in NP_TYPES_MAP:
      raise ValueError("unsupport data type in uTensor: {}".format(np_dtype))
    if idx_fname is None:
      idx_fname = "{}.idx".format(tensor_name.replace(":", "_").replace("/", "_"))

    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    if create_sptr:
      self.template_vars["create_sptr"] = create_sptr
      self.template_vars["sptr_name"] = sptr_name
    self.template_vars["tensor_name"] = tensor_name
    self.template_vars["tensor_shape"] = self._to_shape_str(tensor_shape)
    self.template_vars["dtype"] = NP_TYPES_MAP[np_dtype].tensor_type_str
    self.template_vars["to_eval"] = to_eval

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
  __headers__ = set(['"uTensor/ops/MathOps.hpp"'])

  def __init__(self, inputs, output, np_dtype,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    self.template_vars["in_dtype"] = NP_TYPES_MAP[np_dtype].tensor_type_str
    self.template_vars["out_dtype"] = NP_TYPES_MAP[np_dtype].tensor_type_str
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["to_eval"] = to_eval


class MinOpSnippet(Snippet):
  __template_name__ = "snippets/min_op.cpp"
  __headers__ = set(['"uTensor/ops/MathOps.hpp"'])

  def __init__(self, inputs, output, out_dtype,
               out_shape=None,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["out_shape"] = out_shape
    self.template_vars["to_eval"] = to_eval


class MaxOpSnippet(Snippet):
  __template_name__ = "snippets/max_op.cpp"
  __headers__ = set(['"uTensor/ops/MathOps.hpp"'])

  def __init__(self, inputs, output, out_dtype,
               out_shape=None,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["out_shape"] = out_shape
    self.template_vars["to_eval"] = to_eval

class MaxPoolSnippet(Snippet):
  __template_name__ = "snippets/max_pool_op.cpp"
  __headers__ = set(['"uTensor/ops/NnOps.hpp"'])

  def __init__(self, inputs, output, dtype,
               ksize, strides, padding,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars['ref_count'] = ref_count
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["dtype"] = NP_TYPES_MAP[dtype].tensor_type_str
    _, wind_cols, wind_rows, _ = ksize
    _, col_stride, row_stride, _ = strides
    self.template_vars["wind_cols"] = wind_cols
    self.template_vars["wind_rows"] = wind_rows
    self.template_vars["col_stride"] = col_stride
    self.template_vars["row_stride"] = row_stride
    self.template_vars["padding"] = padding
    self.template_vars["to_eval"] = to_eval


class QuantizedMaxPoolSnippet(Snippet):
  __template_name__ = "snippets/qmax_pool_op.cpp"
  __headers__ = set(['"uTensor/ops/NnOps.hpp"'])

  def __init__(self, inputs, outputs, dtype,
               ksize, strides, padding,
               ref_counts=None,
               to_eval=False):
    Snippet.__init__(self)
    if ref_counts is None:
      ref_counts = []
    if ref_counts:
      err_msg = ("incorrect number of ref_counts and outputs: {}, {}"
                 .format(ref_counts, outputs))
      assert len(ref_counts) == len(outputs), err_msg
      self.template_vars['ref_counts'] = ref_counts
    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["dtype"] = NP_TYPES_MAP[dtype].tensor_type_str
    _, wind_cols, wind_rows, _ = ksize
    _, col_stride, row_stride, _ = strides
    self.template_vars["wind_cols"] = wind_cols
    self.template_vars["wind_rows"] = wind_rows
    self.template_vars["col_stride"] = col_stride
    self.template_vars["row_stride"] = row_stride
    self.template_vars["padding"] = padding
    self.template_vars["to_eval"] = to_eval


class ArgMaxOpSnippet(Snippet):
  __template_name__ = "snippets/argmax_op.cpp"
  __headers__ = set(['"uTensor/ops/MathOps.hpp"'])

  def __init__(self, inputs, output, in_dtype, out_dtype,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["in_dtype"] = NP_TYPES_MAP[in_dtype].tensor_type_str
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["to_eval"] = to_eval


class DequantizeOpSnippet(Snippet):
  __template_name__ = "snippets/dequantize_op.cpp"
  __headers__ = set(['"uTensor/ops/ArrayOps.hpp"'])

  def __init__(self, inputs, output, out_dtype,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["to_eval"] = to_eval

class MatMulOpSnippet(Snippet):
  __template_name__ = "snippets/matmul_op.cpp"
  __headers__ = set(['"uTensor/ops/MatrixOps.hpp"'])

  def __init__(self, inputs, output, x_dtype, w_dtype, out_dtype,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars['ref_count'] = ref_count
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["x_dtype"] = NP_TYPES_MAP[x_dtype].tensor_type_str
    self.template_vars["w_dtype"] = NP_TYPES_MAP[w_dtype].tensor_type_str
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["to_eval"] = to_eval


class QuantizedMatMulOpSnippet(Snippet):
  __template_name__ = "snippets/qmatmul_op.cpp"
  __headers__ = set(['"uTensor/ops/MatrixOps.hpp"'])

  def __init__(self, inputs, outputs, x_dtype, w_dtype, out_dtype,
               ref_counts=None,
               to_eval=False):
    Snippet.__init__(self)
    if ref_counts is None:
      ref_counts = []
    # FIXME: hack on different arguments order between tensorflow and uTensor
    inputs = _permute_args(inputs, [0, 2, 3, 1, 4, 5])
    if ref_counts:
      err_msg = ("incorrect number of ref_counts and outputs: {}, {}"
                 .format(ref_counts, outputs))
      assert len(ref_counts) == len(outputs), err_msg
      self.template_vars['ref_counts'] = ref_counts
    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["x_dtype"] = NP_TYPES_MAP[x_dtype].tensor_type_str
    self.template_vars["w_dtype"] = NP_TYPES_MAP[w_dtype].tensor_type_str
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["to_eval"] = to_eval


class QuantizedAddOpSnippet(Snippet):
  __template_name__ = "snippets/qadd_op.cpp"
  __headers__ = set(['"uTensor/ops/MathOps.hpp"'])

  def __init__(self, inputs, outputs, x_dtype, w_dtype, out_dtype,
               ref_counts=None,
               to_eval=False):
    Snippet.__init__(self)
    if ref_counts is None:
      ref_counts = []
    # hack on different arguments order between tensorflow and uTensor
    inputs = _permute_args(inputs, [0, 2, 3, 1, 4, 5])
    if ref_counts:
      err_msg = ("incorrect number of ref_counts and outputs: {}, {}"
                 .format(ref_counts, outputs))
      assert len(ref_counts) == len(outputs), err_msg
      self.template_vars['ref_counts'] = ref_counts

    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["x_dtype"] = NP_TYPES_MAP[x_dtype].tensor_type_str
    self.template_vars["w_dtype"] = NP_TYPES_MAP[w_dtype].tensor_type_str
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["to_eval"] = to_eval


class QuantizeV2OpSnippet(Snippet):
  __template_name__ = "snippets/quantV2_op.cpp"
  __headers__ = set(['"uTensor/ops/ArrayOps.hpp"'])

  def __init__(self, inputs, outputs, out_dtype,
               ref_counts=None,
               to_eval=False):
    Snippet.__init__(self)
    if ref_counts is None:
      ref_counts = []
    if ref_counts:
      err_msg = ("incorrect number of ref_counts and outputs: {}, {}"
                 .format(ref_counts, outputs))
      assert len(ref_counts) == len(outputs), err_msg
      self.template_vars["ref_counts"] = ref_counts
    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["to_eval"] = to_eval


class ReluOpSnippet(Snippet):
  __template_name__ = "snippets/relu_op.cpp"
  __headers__ = set(['"uTensor/ops/NnOps.hpp"'])

  def __init__(self, inputs, output, in_dtype, out_dtype,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["in_dtype"] = NP_TYPES_MAP[in_dtype].tensor_type_str
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["to_eval"] = to_eval

class QuantizedReluOpSnippet(Snippet):
  __template_name__ = "snippets/qrelu_op.cpp"
  __headers__ = set(['"uTensor/ops/NnOps.hpp"'])

  def __init__(self, inputs, outputs, in_dtype, out_dtypes, qout_dtype,
               ref_counts=None,
               to_eval=False):
    Snippet.__init__(self)
    if ref_counts is None:
      ref_counts = []
    if ref_counts:
      err_msg = ("incorrect number of ref_counts and outputs: {}, {}"
                 .format(ref_counts, outputs))
      assert len(ref_counts) == len(outputs), err_msg
    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["in_dtype"] = NP_TYPES_MAP[in_dtype].tensor_type_str
    self.template_vars["out_dtypes"] = [NP_TYPES_MAP[out_dtype].tensor_type_str for out_dtype in out_dtypes]
    self.template_vars["qout_dtype"] = NP_TYPES_MAP[qout_dtype].tensor_type_str
    self.template_vars["ref_counts"] = ref_counts
    self.template_vars["to_eval"] = to_eval


class RequantizationRangeOpSnippet(Snippet):
  __template_name__ = "snippets/requant_range_op.cpp"
  __headers__ = set(['"uTensor/ops/MathOps.hpp"'])

  def __init__(self, inputs, outputs, out_dtype,
               ref_counts=None,
               to_eval=False):
    Snippet.__init__(self)
    if ref_counts is None:
      ref_counts = []
    if ref_counts:
      err_msg = ("incorrect number of ref_counts and outputs: {}, {}"
                 .format(ref_counts, outputs))
      assert len(ref_counts) == len(outputs), err_msg
    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["ref_counts"] = ref_counts
    self.template_vars["to_eval"] = to_eval


class RequantizeOpSnippet(Snippet):
  __template_name__ = "snippets/requant_op.cpp"
  __headers__ = set(['"uTensor/ops/MathOps.hpp"'])

  def __init__(self, inputs, outputs, qout_dtype, range_dtype,
               ref_counts=None,
               to_eval=False):
    """qout_dtype: Tout
    range_dtype: T2
    input_dtype: T1
    """
    Snippet.__init__(self)
    if ref_counts is None:
      ref_counts = []
    if ref_counts:
      err_msg = ("incorrect number of ref_counts and outputs: {}, {}"
                 .format(ref_counts, outputs))
      assert len(ref_counts) == len(outputs), err_msg
    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["qout_dtype"] = NP_TYPES_MAP[qout_dtype].tensor_type_str
    self.template_vars["range_dtype"] = NP_TYPES_MAP[range_dtype].tensor_type_str
    self.template_vars["ref_counts"] = ref_counts
    self.template_vars["to_eval"] = to_eval


class StridedSliceOpSnippet(Snippet):
  __template_name__ = "snippets/strided_slice_op.cpp"
  __headers__ = set(['"uTensor/ops/ArrayOps.hpp"'])

  def __init__(self, inputs, output, dtype, out_dtype,
               begin_mask, ellipsis_mask, end_mask, new_axis_mask, shrink_axis_mask,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    self.template_vars["dtype"] = NP_TYPES_MAP[dtype].tensor_type_str
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["begin_mask"] = begin_mask
    self.template_vars["ellipsis_mask"] = ellipsis_mask
    self.template_vars["end_mask"] = end_mask
    self.template_vars["new_axis_mask"] = new_axis_mask
    self.template_vars["shrink_axis_mask"] = shrink_axis_mask
    self.template_vars["to_eval"] = to_eval

class PackOpSnippet(Snippet):
  __template_name__ = "snippets/pack_op.cpp"
  __headers__ = set(['"uTensor/ops/ArrayOps.hpp"'])

  def __init__(self, inputs, output, dtype, out_dtype, N, axis,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    self.template_vars["dtype"] = NP_TYPES_MAP[dtype].tensor_type_str
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["N"] = N
    self.template_vars["axis"] = axis
    self.template_vars["to_eval"] = to_eval

class ShapeOpSnippet(Snippet):
  __template_name__ = "snippets/shape_op.cpp"
  __headers__ = set(['"uTensor/ops/ArrayOps.hpp"'])

  def __init__(self, inputs, output, out_dtype,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["to_eval"] = to_eval

class SoftmaxOpSnippet(Snippet):
  __template_name__ = "snippets/softmax_op.cpp"
  __headers__ = set(['"uTensor/ops/NnOps.hpp"'])

  def __init__(self, inputs, output, out_dtype,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["to_eval"] = to_eval


class ReshapeOpSnippet(Snippet):
  __template_name__ = "snippets/reshape_op.cpp"
  __headers__ = set(['"uTensor/ops/ArrayOps.hpp"'])

  def __init__(self, inputs, output, dtype,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    self.template_vars["dtype"] = NP_TYPES_MAP[dtype].tensor_type_str
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["to_eval"] = to_eval


class QuantizedReshapeOpSnippet(Snippet):
  __template_name__ = "snippets/qreshape_op.cpp"
  __headers__ = set(['"uTensor/ops/ArrayOps.hpp"'])

  def __init__(self, inputs, outputs,
               ref_counts=None,
               to_eval=False):
    Snippet.__init__(self)
    if ref_counts:
      self.template_vars["ref_counts"] = ref_counts
    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["to_eval"] = to_eval

class CMSISNNFCOpSnippet(Snippet):
  __template_name__ = "snippets/cmsis_nn_fc_op.cpp"
  __headers__ = set(['"uTensor/ops/cmsis_ops/FullyConnectedOps.hpp"'])

  def __init__(self, inputs, output, in_dtypes, out_dtype,
               ref_counts=None,
               to_eval=False):
    Snippet.__init__(self)
    if ref_counts:
      self.template_vars["ref_counts"] = ref_counts
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["in_dtypes"] = in_dtypes
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["to_eval"] = to_eval

class Conv2DOpSnippent(Snippet):
  __template_name__ = "snippets/conv2d_op.cpp"
  __headers__ = set(['"uTensor/ops/MatrixOps.hpp"'])

  def __init__(self, inputs, output, strides, padding,
               in_dtype, filter_dtype, out_dtype,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["in_dtype"] = NP_TYPES_MAP[in_dtype].tensor_type_str
    self.template_vars["filter_dtype"] = NP_TYPES_MAP[filter_dtype].tensor_type_str
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["strides"] = strides
    self.template_vars["padding"] = padding
    self.template_vars["to_eval"] = to_eval

class Conv2DQuantOpSnippent(Snippet):
  __template_name__ = "snippets/qconv2d_op.cpp"
  __headers__ = set(['"uTensor/ops/MatrixOps.hpp"'])

  def __init__(self, inputs, outputs, strides, padding,
               in_dtype, filter_dtype, out_dtypes,
               ref_counts=None,
               to_eval=False):
    Snippet.__init__(self)
    if ref_counts is None:
      ref_counts = []
    if ref_counts:
      err_msg = ("incorrect number of ref_counts and outputs: {}, {}"
                 .format(ref_counts, outputs))
      assert len(ref_counts) == len(outputs), err_msg
    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["in_dtype"] = NP_TYPES_MAP[in_dtype].tensor_type_str
    self.template_vars["filter_dtype"] = NP_TYPES_MAP[filter_dtype].tensor_type_str
    self.template_vars["out_dtypes"] = [NP_TYPES_MAP[out_dtype].tensor_type_str for out_dtype in out_dtypes]
    self.template_vars["strides"] = strides
    self.template_vars["padding"] = padding
    self.template_vars["ref_counts"] = ref_counts
    self.template_vars["to_eval"] = to_eval

class Uint8Q7OriginSnippet(Snippet):
  __template_name__ = "snippets/cmsis_uint8q7origin_op.cpp"
  __headers__ = set(['"uTensor/ops/cmsis_ops/supportOps.hpp"'])

  def __init__(self, inputs, output,
               ref_count=0,
               to_eval=False):
    Snippet.__init__(self)
    if ref_count:
      self.template_vars["ref_count"] = ref_count
    self.template_vars["inputs"] = inputs
    self.template_vars["output"] = output
    self.template_vars["to_eval"] = to_eval

class QuantRangeForMultiplicationSnippet(Snippet):
  __template_name__ = "snippets/quant_range_for_multiplication_op.cpp"
  __headers__ = set(['"uTensor/ops/cmsis_ops/supportOps.hpp"'])

  def __init__(self, inputs, outputs, out_dtype,
               ref_counts=None,
               to_eval=False):
    Snippet.__init__(self)
    if ref_counts is None:
      ref_counts = []
    if ref_counts:
      err_msg = ("incorrect number of ref_counts and outputs: {}, {}"
                 .format(ref_counts, outputs))
      assert len(ref_counts) == len(outputs), err_msg
      self.template_vars['ref_counts'] = ref_counts
    self.template_vars["inputs"] = inputs
    self.template_vars["outputs"] = outputs
    self.template_vars["out_dtype"] = NP_TYPES_MAP[out_dtype].tensor_type_str
    self.template_vars["to_eval"] = to_eval

class CommentSnippet(Snippet):
  __template_name__ = "snippets/comments.cpp"
  __headers__ = set([])

  def __init__(self, comments):
    Snippet.__init__(self)
    self.template_vars["comments"] = comments


class ContextHeaderSnippet(Snippet):
  __template_name__ = "snippets/get_ctx.hpp"
  __headers__ = set(['"uTensor/core/context.hpp"', '"uTensor/core/tensor.hpp"'])

  def __init__(self, guard_name, graph_name, placeholders=None):
    Snippet.__init__(self)
    if placeholders is None:
      placeholders = []
    self.template_vars["header_guard"] = "_{}_H".format(guard_name.upper())
    self.template_vars["graph_name"] = graph_name
    self.template_vars["placeholders"] = placeholders

class WeightSnippet(Snippet):
  __template_name__ = "snippets/weight_snippet.hpp"
  __headers__ = set([])

  def __init__(self, inline_name, type, shape, value):
      Snippet.__init__(self)
      length = np.prod(shape)
      self.template_vars['type'] =  NP_TYPES_MAP[type].tensor_type_str 
      self.template_vars['value'] = value
      self.template_vars['length'] = int(length) 
      self.template_vars['inline_name'] = inline_name 


class ContextGlobalArrayContainer(SnippetContainerBase):
  __template_name__ = "containers/weight_header.hpp"
  __headers__ = set([])

  def __init__(self, snippets=None):
    SnippetContainerBase.__init__(self, snippets)


class ContextSnippetsContainer(SnippetContainerBase):
  __template_name__ = "containers/get_ctx.cpp"
  __headers__ = set([])

  def __init__(self,
               graph_name, ctx_header_name, ctx_weightheader_name,
               snippets=None, placeholders=None, ref_counts=None):
    SnippetContainerBase.__init__(self, snippets)
    if placeholders is None:
      placeholders = []
    if ref_counts is None:
      ref_counts = []
    if ref_counts:
      err_msg = ("incorrect number of ref_counts and outputs: {}, {}"
                 .format(ref_counts, placeholders))
      assert len(ref_counts) == len(placeholders), err_msg
    self.template_vars["graph_name"] = graph_name
    self.template_vars["placeholders"] = placeholders
    self.template_vars["ref_counts"] = ref_counts
    self.add_header('"{}"'.format(ctx_header_name))
    self.add_header('"{}"'.format(ctx_weightheader_name))
