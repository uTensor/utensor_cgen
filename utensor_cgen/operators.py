# -*- coding:utf8 -*-
from .snippets import *  # pylint: disable=W0401,W0614


class _Operator(object):
  def __init__(self):
    self.name = ""
    self._snippet = None

  @property
  def snippet(self):
    return self._snippet


class _AddOperator(_Operator):
  def __init__(self, op_info, ref_counts, to_eval):
    _Operator.__init__(self)
    inputs = [tname for tname, _, _ in op_info.input_tensor]
    output, _, _ = op_info.output_tensor[0]
    _, tf_dtype, _ = op_info.input_tensor[0]
    ref_count = ref_counts[0]
    self._snippet = AddOpSnippet(inputs, output, tf_dtype, ref_count, to_eval)


class _ArgMaxOperator(_Operator):
  def __init__(self, op_info, ref_counts, to_eval):
    _Operator.__init__(self)
    inputs = [tname for tname, _, _ in op_info.input_tensor]
    output, out_dtype, _ = op_info.output_tensor[0]
    _, in_dtype, _ = op_info.input_tensor[0]
    ref_count = ref_counts[0]
    self._snippet = ArgMaxOpSnippet(inputs, output, in_dtype, out_dtype, ref_count, to_eval)


class _DequantizeOperator(_Operator):
  def __init__(self, op_info, ref_counts, to_eval):
    _Operator.__init__(self)
    inputs = [tname for tname, _, _ in op_info.input_tensor]
    output, out_dtype, _ = op_info.output_tensor[0]
    ref_count = ref_counts[0]
    self._snippet = DequantizeOpSnippet(inputs, output, out_dtype, ref_count, to_eval)


class _MaxOperator(_Operator):
  def __init__(self, op_info, ref_counts, to_eval):
    _Operator.__init__(self)
    inputs = [tname for tname, _, _ in op_info.input_tensor]
    output, out_dtype, out_shape = op_info.output_tensor[0]
    # FIXME automatic alloc for uTensor fail
    if not out_shape:
      out_shape = [1]
    ref_count = ref_counts[0]
    self._snippet = MaxOpSnippet(inputs, output, out_dtype, out_shape, ref_count, to_eval)


class _MinOperator(_Operator):
  def __init__(self, op_info, ref_counts, to_eval):
    _Operator.__init__(self)
    inputs = [tname for tname, _, _ in op_info.input_tensor]
    output, out_dtype, out_shape = op_info.output_tensor[0]
    # FIXME automatic alloc for uTensor fail
    if not out_shape:
      out_shape = [1]
    ref_count = ref_counts[0]
    self._snippet = MinOpSnippet(inputs, output, out_dtype, out_shape, ref_count, to_eval)


class _QuantizeV2Operator(_Operator):
  def __init__(self, op_info, ref_counts, to_eval):
    _Operator.__init__(self)
    inputs = [tname for tname, _, _ in op_info.input_tensor]
    outputs = [tname for tname, _, _ in op_info.output_tensor]
    _, out_dtype, _ = op_info.output_tensor[0]
    self._snippet = QuantizeV2OpSnippet(inputs, outputs, out_dtype, ref_counts, to_eval)


class _QuantizedMatMulOperator(_Operator):
  def __init__(self, op_info, ref_counts, to_eval):
    _Operator.__init__(self)
    inputs = [tname for tname, _, _ in op_info.input_tensor]
    outputs = [tname for tname, _, _ in op_info.output_tensor]
    _, x_dtype, _ = op_info.input_tensor[0]
    _, w_dtype, _ = op_info.input_tensor[1]
    _, out_dtype, _ = op_info.output_tensor[0]
    self._snippet = QuantizedMatMulOpSnippet(inputs, outputs, x_dtype, w_dtype, out_dtype, ref_counts, to_eval)


class _QuantizedReluOperator(_Operator):
  def __init__(self, op_info, ref_counts, to_eval):
    _Operator.__init__(self)
    inputs = [tname for tname, _, _ in op_info.input_tensor]
    outputs = [tname for tname, _, _ in op_info.output_tensor]
    _, in_dtype, _ = op_info.input_tensor[0]
    _, qout_dtype, _ = op_info.output_tensor[0]
    out_dtypes = [t[1] for t in op_info.output_tensor[1:]]
    self._snippet = QuantizedReluOpSnippet(inputs, outputs, in_dtype, out_dtypes, qout_dtype, ref_counts, to_eval)


class _RequantizationRangeOperator(_Operator):
  def __init__(self, op_info, ref_counts, to_eval):
    _Operator.__init__(self)
    inputs = [tname for tname, _, _ in op_info.input_tensor]
    outputs = [tname for tname, _, _ in op_info.output_tensor]
    _, out_dtype, _ = op_info.output_tensor[0]
    self._snippet = RequantizationRangeOpSnippet(inputs, outputs, out_dtype, ref_counts, to_eval)


class _RequantizeOperator(_Operator):
  def __init__(self, op_info, ref_counts, to_eval):
    _Operator.__init__(self)
    inputs = [tname for tname, _, _ in op_info.input_tensor]
    outputs = [tname for tname, _, _ in op_info.output_tensor]
    _, qout_dtype, _ = op_info.output_tensor[0]
    _, range_dtype, _ = op_info.output_tensor[1]
    self._snippet = RequantizeOpSnippet(inputs, outputs, qout_dtype, range_dtype, ref_counts, to_eval)


class _ReshapeOperator(_Operator):
  def __init__(self, op_info, ref_counts, to_eval):
    _Operator.__init__(self)
    inputs = [tname for tname, _, _ in op_info.input_tensor]
    output, _, _ = op_info.output_tensor[0]
    ref_count = ref_counts[0]
    self._snippet = ReshapeOpSnippet(inputs, output, ref_count, to_eval)


class OperatorFactory():
  # Can easily do something smarter
  _operators = {"Add": _AddOperator,
                "ArgMax": _ArgMaxOperator,
                "Dequantize": _DequantizeOperator,
                "Max": _MaxOperator,
                "Min": _MinOperator,
                "QuantizeV2": _QuantizeV2Operator,
                "QuantizedMatMul": _QuantizedMatMulOperator,
                "QuantizedRelu": _QuantizedReluOperator,
                "RequantizationRange": _RequantizationRangeOperator,
                "Requantize": _RequantizeOperator,
                "Reshape": _ReshapeOperator}

  def createOperatorSnippet(self, op_info, ref_counts, to_eval):
    op_type = op_info.op_type
    if op_type not in self._operators:
      err_msg = "unsupported op type in uTensor: {}".format(op_type)
      raise ValueError(err_msg)

    op = self._operators[op_type](op_info, ref_counts, to_eval)  # Create desired object
    return op.snippet  # Ops know how to create their snippets
