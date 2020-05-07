from typing import Hashable

from utensor_cgen.backend.utensor.snippets._types import NP_TYPES_MAP
from utensor_cgen.backend.utensor.snippets.rearch import *
from utensor_cgen.utils import must_return_type

from ._base import OperatorFactory, _Operator


@OperatorFactory.register
class _AddOperator(_Operator):
  op_type = 'AddOperator'

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return AddOpEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
    )


@OperatorFactory.register
class _ReshapeOperator(_Operator):
  op_type = "ReshapeOperator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return ReshahpeEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
    )


@OperatorFactory.register
class _MatmulOperator(_Operator):
  op_type = 'MatrixMultOperator'

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return MatrixMultEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
    )


@OperatorFactory.register
class _ArgMinOperator(_Operator):
  op_type = "ArgMinOperator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return ArgMinEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
    )


@OperatorFactory.register
class _ArgMaxOperator(_Operator):
  op_type = "ArgMaxOperator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return ArgMaxEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
    )


@OperatorFactory.register
class _QuantizeOperator(_Operator):
  op_type = "QuantizeOperator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.out_dtypes[0], self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=['TFLM'],
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return QuantizeEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.out_dtypes[0], self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=['TFLM'],
    )


@OperatorFactory.register
class _DequantizeOperator(_Operator):
  op_type = "DequantizeOperator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.out_dtypes[0], self.in_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=['TFLM'],
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return DequantizeEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.out_dtypes[0], self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=['TFLM'],
    )


@OperatorFactory.register
class _ReLUOperator(_Operator):
  op_type = "ReLUOperator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return ReLUEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
    )


@OperatorFactory.register
class _ReLU6Operator(_Operator):
  op_type = "ReLU6Operator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return ReLU6EvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
    )


@OperatorFactory.register
class _MinOperator(_Operator):
  op_type = 'MinOperator'

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return MinEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
    )


@OperatorFactory.register
class _MaxOperator(_Operator):
  op_type = 'MaxOperator'

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return MaxEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
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
    stride_str = "{{ {} }}".format(", ".join(map(str, strides)))
    ksize_str = "{{ {} }}".format(", ".join(map(str, ksize)))
    return (stride_str, ksize_str, padding)


@OperatorFactory.register
class _MaxPoolOperator(_PoolingOperatorMixin, _Operator):
  op_type = 'MaxPoolOperator'

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
    )
  
  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return MaxPoolEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
    )


@OperatorFactory.register
class _MinPoolOperator(_PoolingOperatorMixin, _Operator):
  op_type = 'MinPoolOperator'

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.in_dtypes[0]],
      op_var_name=op_var_name,
    )
  
  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return MinPoolEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.in_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
    )


@OperatorFactory.register
class _QuantDWSConvOperator(_Operator):
  namespaces = ('TFLM',)
  op_type = "DepthwiseSeparableConvOperator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.out_dtypes[0]],
      op_var_name=op_var_name,
      nested_namespaces=self.namespaces,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return QuantDepthwiseSeperateConvOpEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.out_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
      nested_namespaces=self.namespaces,
    )


@OperatorFactory.register
class _DWSConvOperator(_Operator):
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
    padding = op_info.op_attr['Padding'] == 1 and "VALID" or "SAME"
    strides_str = ','.join(map(str, strides))
    return ("{{ {} }}".format(strides_str), padding)

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.out_dtypes[0]],
      op_var_name=op_var_name,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return DepthwiseSeperateConvOpEvalSnippet(
      op_info=op_info,
      templ_dtypes=[self.out_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
    )


@OperatorFactory.register
class _QuantizedFullyConnectedOperator(_Operator):
  op_type = "QuantizedFullyConnectedOperator"

  def get_declare_snippet(self, op_var_name, tensor_var_map):
    return DeclareOpSnippet(
      op=self,
      templ_dtypes=[self.out_dtypes[0]],
      op_var_name=op_var_name,
    )

  def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
    return QuantizedFullyConnectedSnippet(
      op_info=op_info,
      templ_dtypes=[self.out_dtypes[0]],
      op_name=op_var_name,
      tensor_var_map=tensor_var_map,
    )
