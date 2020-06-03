import re
from typing import Hashable

from utensor_cgen.backend.utensor.snippets._types import NP_TYPES_MAP
from utensor_cgen.backend.utensor.snippets.rearch import *
from utensor_cgen.utils import must_return_type

from ._base import OperatorFactory, _Operator


def _c_arr_str(iterable):
  return "{{ {} }}".format(
    ", ".join(map(str, iterable))
  )


@OperatorFactory.register
class _AddOperator(_Operator):
  namespaces = ('ReferenceOperators',)
  op_type = 'AddOperator'

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return AddOpEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


@OperatorFactory.register
class _ReshapeOperator(_Operator):
  namespaces = ('ReferenceOperators',)
  op_type = "ReshapeOperator"

  @classmethod
  @must_return_type(Hashable)
  def get_constructor_parameters(cls, op_info):
    new_shape = op_info.op_attr['new_shape']
    return (_c_arr_str(new_shape),)

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return ReshahpeEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


@OperatorFactory.register
class _MatmulOperator(_Operator):
  namespaces = ('ReferenceOperators',)
  op_type = 'MatrixMultOperator'

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return MatrixMultEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


@OperatorFactory.register
class _ArgMinOperator(_Operator):
  namespaces = ('ReferenceOperators',)
  op_type = "ArgMinOperator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return ArgMinEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


@OperatorFactory.register
class _ArgMaxOperator(_Operator):
  namespaces = ('ReferenceOperators',)
  op_type = "ArgMaxOperator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return ArgMaxEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


@OperatorFactory.register
class _QuantizeOperator(_Operator):
  namespaces = ('TflmSymQuantOps',)
  op_type = "QuantizeOperator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.out_dtypes[0], self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return QuantizeEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.out_dtypes[0], self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


@OperatorFactory.register
class _DequantizeOperator(_Operator):
  namespaces = ('TflmSymQuantOps',)
  op_type = "DequantizeOperator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.out_dtypes[0], self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return DequantizeEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.out_dtypes[0], self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


@OperatorFactory.register
class _ReLUOperator(_Operator):
  namespaces = ('ReferenceOperators',)
  op_type = "ReLUOperator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return ReLUEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


@OperatorFactory.register
class _ReLU6Operator(_Operator):
  namespaces = ('ReferenceOperators',)
  op_type = "ReLU6Operator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return ReLU6EvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


@OperatorFactory.register
class _MinOperator(_Operator):
  namespaces = ('ReferenceOperators',)
  op_type = 'MinOperator'

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return MinEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


@OperatorFactory.register
class _MaxOperator(_Operator):
  namespaces = ('ReferenceOperators',)
  op_type = 'MaxOperator'

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return MaxEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


class _PoolingOperatorMixin(object):

  @classmethod
  @must_return_type(Hashable)
  def get_constructor_parameters(cls, op_info):
    if op_info.ugraph.lib_name == "tensorflow":
      strides = op_info.op_attr['strides'].value.ints_value
      ksize = op_info.op_attr['ksize'].value.ints_value[1:3]
      padding = op_info.op_attr['padding'].value.decode('utf8')
    elif op_info.ugraph.lib_name == 'tflite':
      strides = [
        1,
        op_info.op_attr['StrideW'],
        op_info.op_attr['StrideH'],
        1,
      ]
      ksize = [
        op_info.op_attr['FilterWidth'],
        op_info.op_attr['FilterHeight'],
      ]
      padding = op_info.op_attr['Padding'] == 1 and "VALID" or "SAME"
    else:
      raise RuntimeError("dont know to to get constructor signature")
    stride_str = _c_arr_str(strides)
    ksize_str = _c_arr_str(ksize)
    return (ksize_str, stride_str, padding)


@OperatorFactory.register
class _MaxPoolOperator(_PoolingOperatorMixin, _Operator):
  namespaces = ('ReferenceOperators',)
  op_type = 'MaxPoolOperator'

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )
  
  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return MaxPoolEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


@OperatorFactory.register
class _MinPoolOperator(_PoolingOperatorMixin, _Operator):
  namespaces = ('ReferenceOperators',)
  op_type = 'MinPoolOperator'

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )
  
  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return MinPoolEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


class _CommonParams(_Operator):
  _PADDING_MAP = {
    0: "UNKNOWN",
    1: "VALID",
    2: "SAME"
  }
  _ACTIVATION_MAP = {
    '0': 'TFLM::TfLiteFusedActivation::kTfLiteActNone',
    '1': 'TFLM::TfLiteFusedActivation::kTfLiteActRelu',
    '2': 'TFLM::TfLiteFusedActivation::kTfLiteActRelu1',
    '3': 'TFLM::TfLiteFusedActivation::kTfLiteActRelu6',
    '4': 'TFLM::TfLiteFusedActivation::kTfLiteActTanh',
    '5': 'TFLM::TfLiteFusedActivation::kTfLiteActSignBit',
    '6': 'TFLM::TfLiteFusedActivation::kTfLiteActSigmoid',
  }
  _ACTIVATION_STR_PATTERN = re.compile(r'^(\d+) \(\w+\)$')


@OperatorFactory.register
class _Conv2dOperator(_CommonParams):
  namespaces = ('ReferenceOperators',)
  op_type = 'Conv2dOperator'

  @classmethod
  @must_return_type(Hashable)
  def get_constructor_parameters(cls, op_info):
    padding = cls._PADDING_MAP[op_info.op_attr['Padding']]
    stride_width = op_info.op_attr['StrideW']
    stride_hight = op_info.op_attr['StrideH']
    return (
      _c_arr_str([1, stride_hight, stride_width, 1]),
      padding,
    )

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.out_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return Conv2dOpEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.out_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


@OperatorFactory.register
class _QuantDWSConvOperator(_CommonParams):
  namespaces = ('TflmSymQuantOps',)
  op_type = "DepthwiseSeparableConvOperator"

  @classmethod
  @must_return_type(Hashable)
  def get_constructor_parameters(cls, op_info):
    padding = cls._PADDING_MAP[op_info.op_attr['Padding']]
    stride_width = op_info.op_attr['StrideW']
    stride_hight = op_info.op_attr['StrideH']
    depth_multiplier = op_info.op_attr['DepthMultiplier']
    activation_idx = cls._ACTIVATION_STR_PATTERN.match(
      op_info.op_attr['FusedActivationFunction']
    ).group(1)
    activation = cls._ACTIVATION_MAP[activation_idx]
    dilation_width_factor = op_info.op_attr['DilationWFactor']
    dilation_height_factor = op_info.op_attr['DilationHFactor']
    return (
      _c_arr_str([stride_width, stride_hight]),
      padding,
      depth_multiplier,
      _c_arr_str([dilation_width_factor, dilation_height_factor]),
      activation,
    )

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.out_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return QuantDepthwiseSeperateConvOpEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.out_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


@OperatorFactory.register
class _DWSConvOperator(_CommonParams):
  namespaces = ('ReferenceOperators',)
  op_type = "DepthwiseSeparableConvOperator"

  @classmethod
  @must_return_type(Hashable)
  def get_constructor_parameters(cls, op_info):
    strides = [
        1,
        op_info.op_attr['StrideW'],
        op_info.op_attr['StrideH'],
        1,
      ]
    padding = cls._PADDING_MAP[op_info.op_attr['Padding']]
    strides_str = ','.join(map(str, strides))
    return ("{{ {} }}".format(strides_str), padding)

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.out_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return DepthwiseSeperateConvOpEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.out_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


@OperatorFactory.register
class _QuantizedFullyConnectedOperator(_CommonParams):
  namespaces = ('TflmSymQuantOps',)
  op_type = "FullyConnectedOperator"

  @classmethod
  @must_return_type(Hashable)
  def get_constructor_parameters(cls, op_info):
    activation_idx = cls._ACTIVATION_STR_PATTERN.match(
      op_info.op_attr['FusedActivationFunction']
    ).group(1)
    activation = cls._ACTIVATION_MAP[activation_idx]
    return (activation,)

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.out_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=type(self).namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return QuantizedFullyConnectedSnippet(
      op_info=op_info,
      templ_dtypes=[self.out_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=type(self).namespaces,
    )


class _MissingOperator(_Operator):
  op_type = "_MissingOperator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return None
  
  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return MissingOpEvalSnippet(op_info, tensor_var_map)

OperatorFactory._operators[_MissingOperator.op_type] = _MissingOperator
