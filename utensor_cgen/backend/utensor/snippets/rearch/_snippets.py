import re

import numpy as np

from utensor_cgen.backend.utensor.snippets._base import Snippet, SnippetBase
from utensor_cgen.backend.utensor.snippets._types import (NP_TYPES_MAP,
                                                          UTENSOR_TYPES_MAP)

__all__ = [
  "DeclareRomTensorSnippet",
  "DeclareRamTensorSnippet",
  "FreeTensorSnippet",
  "DeclareOpSnippet",
  "Conv2dOpEvalSnippet",
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
  "MissingOpEvalSnippet",
  "BatchNormSnippet",
  "TimeSlotContainer",
  "MulOpEvalSnippet",
  "SubOpEvalSnippet",
  "ConvOpEvalSnippet",
  "MeanOpEvalSnippet",
  "SoftmaxOpEvalSnippet",
  "SigmoidOpEvalSnippet",
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
      quant_params['is_per_tensor'] = zeros.size == 1
      quant_params['zero_point'] = {
        'value': zeros,
        # fixing the type to int32_t for the design of runtime
        'type_str': 'int32_t', #NP_TYPES_MAP[zeros.dtype].tensor_type_str
      }
      quant_params['scale'] = {
        'value': scales,
        'type_str': 'float'
      }
    return quant_params


# op declare snippets
class _DeclareTensorBase(_SnippetBase):

  def __init__(self, tensor_info, tensor_var):
    _SnippetBase.__init__(self)
    quant_params = self.get_quant_param(tensor_info)
    self.template_vars['quant_params'] = quant_params


class DeclareRomTensorSnippet(_DeclareTensorBase):
  __template_name__ = 'snippets/rearch/declare_rom_tensor.cpp'

  def __init__(self, tensor_info, tensor_var, buffer_var, static=False):
    _DeclareTensorBase.__init__(self, tensor_info, tensor_var)
    self.template_vars['tensor_var'] = tensor_var
    self.template_vars['shape'] = tensor_info.shape or [1]
    self.template_vars['buffer_var'] = buffer_var
    self.template_vars['static'] = static
    self.template_vars['utensor_dtype'] = UTENSOR_TYPES_MAP[tensor_info.dtype]


class DeclareRamTensorSnippet(_DeclareTensorBase):
  __template_name__ = 'snippets/rearch/declare_ram_tensor.cpp'

  def __init__(self, tensor_info, tensor_var):
    _DeclareTensorBase.__init__(self, tensor_info, tensor_var)
    self.template_vars['tensor_var'] = tensor_var
    self.template_vars['shape'] = tensor_info.shape or [1]
    self.template_vars['utensor_dtype'] = UTENSOR_TYPES_MAP[tensor_info.dtype]


class FreeTensorSnippet(_SnippetBase):
  __template_name__ = 'snippets/rearch/tensor_free.cpp'

  def __init__(self, tensor_var):
    _SnippetBase.__init__(self)
    self.template_vars['tensor_var'] = tensor_var


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


# op eval snippets
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


class Conv2dOpEvalSnippet(OpEvalSnippet):
  __inputs__ = ["in", "filter"]
  __outputs__ = ["out"]


class DepthwiseSeperateConvOpEvalSnippet(OpEvalSnippet):
  __inputs__ = ["in", "depthwise_filter", "pointwise_filter"]
  __outputs__ = ["out"]

class ConvOpEvalSnippet(OpEvalSnippet):
  __inputs__ = ["in", "filter"]
  __outputs__ = ["out"]

class QuantDepthwiseSeperateConvOpEvalSnippet(OpEvalSnippet):
  __inputs__ = ["in", "filter", "bias"]
  __outputs__ = ["out"]


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

class BatchNormSnippet(OpEvalSnippet):
  __inputs__ = ["x", "mean", "variance", "offset", "scale"]
  __outputs__ = ["output"]

class MulOpEvalSnippet(OpEvalSnippet):
  __inputs__ = ['a', 'b']
  __outputs__ = ['c']
class SubOpEvalSnippet(OpEvalSnippet):
  __inputs__ = ['a', 'b']
  __outputs__ = ['c']
class MeanOpEvalSnippet(OpEvalSnippet):
  __inputs__ = ['input', 'axis']
  __outputs__ = ['output']
class SoftmaxOpEvalSnippet(OpEvalSnippet):
  __inputs__ = ['input']
  __outputs__ = ['output']


class MissingOpEvalSnippet(OpEvalSnippet):
  __template_name__ = "snippets/rearch/op_missing.cpp"

  def __init__(self, op_info, tensor_var_map):
    Snippet.__init__(self)

    quant_params_map = {}
    for out_tensor in op_info.output_tensors:
      quant_params = self.get_quant_param(out_tensor)
      quant_params_map[out_tensor.name] = quant_params
    self.template_vars['op_type'] = op_info.op_type
    self.template_vars['input_var_names'] = [
      tensor_var_map[tensor.name] for tensor in op_info.input_tensors
    ]
    self.template_vars['input_tensors'] = op_info.input_tensors[:]
    self.template_vars['out_var_names'] = [
      tensor_var_map[tensor.name] for tensor in op_info.output_tensors
    ]
    self.template_vars['output_tensors'] = op_info.output_tensors[:]
    self.template_vars['quant_params_map'] = quant_params_map
class SigmoidOpEvalSnippet(OpEvalSnippet):
  __inputs__ = ['in']
  __outputs__ = ['out']


class TimeSlotContainer(SnippetBase):
  __template_name__ = 'containers/rearch/time_slot.cpp'
  __headers__ = set(['"uTensor.h"'])

  def __init__(self ):
    SnippetBase.__init__(self)
    self.__headers__ = set(type(self).__headers__)
    self._local_snippets = []

  def add_local_snippets(self, *local_snippets):
    for snippet in local_snippets:
      self._local_snippets.append(snippet)
      self.__headers__.update(
        snippet.__headers__
      )

  def render(self):
    return self.template.render(
      local_snippets=self._local_snippets,
      **self.template_vars
    )


class SimpleContainer(SnippetBase):
  __template_name__ = 'containers/rearch/simple.cpp'
  __headers__ = set(['"uTensor.h"'])

  def __init__(self):
    SnippetBase.__init__(self)
    self.__headers__ = set(type(self).__headers__)
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
