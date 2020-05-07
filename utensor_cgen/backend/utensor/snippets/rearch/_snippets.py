import re

import numpy as np

from utensor_cgen.backend.utensor.snippets._base import Snippet, SnippetBase
from utensor_cgen.backend.utensor.snippets._types import (NP_TYPES_MAP,
                                                          UTENSOR_TYPES_MAP)

__all__ = [
  "DeclareRomTensorSnippet",
  "DeclareRamTensorSnippet",
  "DeclareOpSnippet",
  "DepthwiseSeperateConvOpEvalSnippet",
  "QuantDepthwiseSeperateConvOpEvalSnippet",
  "AddOpEvalSnippet",
  "ReshahpeEvalSnippet",
  "QuantizeEvalSnippet",
  "MatrixMultEvalSnippet",
  "ArgMaxEvalSnippet",
  "ArgMinEvalSnippet",
  "DequantizeEvalSnippet",
  "ReLUEvalSnippet",
  "ReLU6EvalSnippet",
  "MinEvalSnippet",
  "MaxEvalSnippet",
  "MinPoolEvalSnippet",
  "MaxPoolEvalSnippet",
  "QuantizedFullyConnectedSnippet",
  "SimpleContainer",
]

class _SnippetBase(Snippet):
  __headers__ = set(['"uTensor.h"'])

  @staticmethod
  def get_quant_param(tensor_info):
    quant_params = {}
    if 'quantization_zeros' in tensor_info.attributes:
      zeros = tensor_info.attributes['quantization_zeros']
      scales = tensor_info.attributes["quantization_scales"]
      quant_params['zero_point'] = {
        'value': zeros[0],
        'type_str': ['uint8_t', 'int8_t'][zeros.dtype == np.dtype('int8')]
      }
      quant_params['scale'] = {
        'value': scales[0],
        'type_str': 'float'
      }
    return quant_params


class DeclareRomTensorSnippet(_SnippetBase):
  __template_name__ = 'snippets/rearch/declare_rom_tensor.cpp'

  def __init__(self, tensor_info, tensor_var, buffer_var, static=False):
    _SnippetBase.__init__(self)
    self.template_vars['tensor_var'] = tensor_var
    self.template_vars['shape'] = tensor_info.shape or [1]
    self.template_vars['buffer_var'] = buffer_var
    self.template_vars['static'] = static
    self.template_vars['utensor_dtype'] = UTENSOR_TYPES_MAP[tensor_info.dtype]
    self.template_vars['quantize_params'] = self.get_quant_param(tensor_info)


class DeclareRamTensorSnippet(_SnippetBase):
  __template_name__ = 'snippets/rearch/declare_ram_tensor.cpp'

  def __init__(self, tensor_info, tensor_var):
    _SnippetBase.__init__(self)
    self.template_vars['tensor_var'] = tensor_var
    self.template_vars['shape'] = tensor_info.shape or [1]
    self.template_vars['utensor_dtype'] = UTENSOR_TYPES_MAP[tensor_info.dtype]
    self.template_vars['quantize_params'] = self.get_quant_param(tensor_info)


class DeclareOpSnippet(_SnippetBase):
  __template_name__ = 'snippets/rearch/declare_op.cpp'

  def __init__(self, op, templ_dtypes, op_var_name, nested_namespaces=None):
    _SnippetBase.__init__(self)
    if nested_namespaces is None:
      nested_namespaces = []
    else:
      nested_namespaces = list(nested_namespaces)
    op_type = op.op_type
    if templ_dtypes:
      templ_params = ', '.join([NP_TYPES_MAP[dtype].tensor_type_str for dtype in templ_dtypes])
      op_type = '{}<{}>'.format(op_type, templ_params)
    if nested_namespaces:
      op_type = "::".join(nested_namespaces + [op_type])
    self.template_vars['op_type'] = op_type
    self.template_vars['construct_params'] = op.construct_params
    self.template_vars['op_var_name'] = op_var_name


class OpEvalSnippet(_SnippetBase):
  __template_name__ = 'snippets/rearch/eval_op.cpp'
  __inputs__ = []
  __outputs__ = []

  def __init__(self, op_info, templ_dtypes, op_name, tensor_var_map, nested_namespaces=None):
    Snippet.__init__(self)
    if nested_namespaces is None:
      nested_namespaces = []
    else:
      nested_namespaces = list(nested_namespaces)
    input_map = {
      name: tensor_var_map[tensor.name]
      for name, tensor in zip(self.__inputs__, op_info.input_tensors)
    }
    output_map = {
      name: tensor_var_map[tensor.name]
      for name, tensor in zip(self.__outputs__, op_info.output_tensors)
    }
    quantize_params_map = {}
    for tensor_info in op_info.output_tensors:
      quant_param = self.get_quant_param(tensor_info)
      if quant_param:
        tensor_var = tensor_var_map[tensor_info.name]
        quantize_params_map[tensor_var] = quant_param
    if templ_dtypes:
      templ_params = ', '.join([NP_TYPES_MAP[dtype].tensor_type_str for dtype in templ_dtypes])
      op_type = '{}<{}>'.format(op_info.op_type, templ_params)
    else:
      op_type = op_info.op_type
    if nested_namespaces:
      op_type = "::".join(nested_namespaces + [op_type])
    self.template_vars['op_type'] = op_type
    self.template_vars['op_var_name'] = op_name
    self.template_vars['input_map'] = input_map
    self.template_vars['output_map'] = output_map
    self.template_vars['quantize_params_map'] = quantize_params_map


class DepthwiseSeperateConvOpEvalSnippet(OpEvalSnippet):
  __inputs__ = ["in", "depthwise_filter", "pointwise_filter"]
  __outputs__ = ["out"]


class QuantDepthwiseSeperateConvOpEvalSnippet(OpEvalSnippet):
  __template_name__ = 'snippets/rearch/eval_quant_dws_conv_op.cpp'
  __inputs__ = ["in", "filter", "bias"]
  __outputs__ = ["out"]

  _PADDING_MAP = {
    0: "TFLM::TfLitePadding::kTfLitePaddingUnknown",
    1: "TFLM::TfLitePadding::kTfLitePaddingSame",
    2: "TFLM::TfLitePadding::kTfLitePaddingValid"
  }
  _ACTIVATION_MAP = {
    '0': 'kTfLiteActNone',
    '1': 'kTfLiteActRelu',
    '2': 'kTfLiteActRelu1',
    '3': 'kTfLiteActRelu6',
    '4': 'kTfLiteActTanh',
    '5': 'kTfLiteActSignBit',
    '6': 'kTfLiteActSigmoid',
  }
  _ACTIVATION_STR_PATTERN = re.compile(r'^(\d+) \(\w+\)$')

  def __init__(self, op_info, templ_dtypes, op_name, tensor_var_map, nested_namespaces=None):
    OpEvalSnippet.__init__(self, op_info, templ_dtypes, op_name, tensor_var_map, nested_namespaces)
    cls = type(self)
    self.template_vars['padding'] = cls._PADDING_MAP[op_info.op_attr['Padding']]
    self.template_vars['stride_width'] = op_info.op_attr['StrideW']
    self.template_vars['stride_height'] = op_info.op_attr['StrideH']
    self.template_vars['depth_multiplier'] = op_info.op_attr['DepthMultiplier']
    activation_idx = cls._ACTIVATION_STR_PATTERN.match(
      op_info.op_attr['FusedActivationFunction']
    ).group(1)
    self.template_vars['activation'] = cls._ACTIVATION_MAP[activation_idx]
    self.template_vars['dilation_width_factor'] = op_info.op_attr['DilationWFactor']
    self.template_vars['dilation_height_factor'] = op_info.op_attr['DilationHFactor']


class AddOpEvalSnippet(OpEvalSnippet):
  __inputs__ = ['a', 'b']
  __outputs__ = ['c']


class ReshahpeEvalSnippet(OpEvalSnippet):
  __inputs__ = ["input"]
  __outputs__ = ["output"]


class QuantizeEvalSnippet(OpEvalSnippet):
  __inputs__ = ["input"]
  __outputs__ = ["output"]


class MatrixMultEvalSnippet(OpEvalSnippet):
  __inputs__ = ["a", "b"]
  __outputs__ = ["c"]


class ArgMaxEvalSnippet(OpEvalSnippet):
  __inputs__ = ["input"]
  __outputs__ = ["output"]


class ArgMinEvalSnippet(OpEvalSnippet):
  __inputs__ = ["input"]
  __outputs__ = ["output"]


class DequantizeEvalSnippet(OpEvalSnippet):
  __inputs__ = ["a"]
  __outputs__ = ["b"]


class ReLUEvalSnippet(OpEvalSnippet):
  __inputs__ = ["in"]
  __outputs__ = ["out"]


class ReLU6EvalSnippet(OpEvalSnippet):
  __inputs__ = ["in"]
  __outputs__ = ["out"]


class MinEvalSnippet(OpEvalSnippet):
  __inputs__ = ["in"]
  __outputs__ = ["out"]


class MaxEvalSnippet(OpEvalSnippet):
  __inputs__ = ["in"]
  __outputs__ = ["out"]


class MinPoolEvalSnippet(OpEvalSnippet):
  __inputs__ = ["in"]
  __outputs__ = ["out"]


class MaxPoolEvalSnippet(OpEvalSnippet):
  __inputs__ = ["in"]
  __outputs__ = ["out"]


class QuantizedFullyConnectedSnippet(OpEvalSnippet):
  __inputs__ = ["input", "filter", "bias"]
  __outputs__ = ["output"]


class SimpleContainer(SnippetBase):
  __headers__ = set(['"uTensor.h"', "<vector>"])
  __template_name__ = 'containers/rearch/simple.cpp'

  def __init__(self):
    SnippetBase.__init__(self)
    self._declare_local_snippets = []
    self._declare_global_snippets = []
    self._eval_snippests = []

  def add_declare_global_snippets(self, *snippets):
    for snippet in snippets:
      self.__headers__.update(snippet.headers)
      self._declare_global_snippets.append(snippet)
  
  def add_declare_local_snippets(self, *snippets):
    for snippet in snippets:
      self.__headers__.update(snippet.headers)
      self._declare_local_snippets.append(snippet)

  def add_eval_snippets(self, *snippets):
    for snippet in snippets:
      self.__headers__.update(snippet.headers)
      self._eval_snippests.append(snippet)
  
  def add_header(self, header, *headers):
    self._add_header(header)
    for header in headers:
      self._add_header(header)
    return self
  
  def _add_header(self, header):
    if not header.startswith('"') and not header.startswith("<"):
      header = '"{}"'.format(header)
    self.__headers__.add(header)
  
  def render(self):
    return self.template.render(
      declare_global_snippets=self._declare_global_snippets,
      declare_local_snippets=self._declare_local_snippets,
      eval_snippets=self._eval_snippests,
      **self.template_vars
    )
