import re
from copy import deepcopy
from functools import reduce

from utensor_cgen.ir.base import OperationInfo, TensorInfo
from utensor_cgen.utils import prune_graph, topologic_order_graph

from .api import Legalizer
from .base import LegalizerBase
from .utils import _hotfix_reshape


@Legalizer.register
class TFLiteLegalizer(LegalizerBase):
  TARGET = "tflite"

  def legalize_ops(self, ugraph):
    _GraphRewrite.apply(ugraph)
    _OpTypeRename.apply(ugraph)
    ugraph = _hotfix_reshape(ugraph)
    return ugraph
  
  def legalize_dtype(self, ugraph):
    return ugraph

  @classmethod
  def register_op_rename(cls, old_name, new_name):
    _OpTypeRename._OPTYPE_RENAME_MAP[old_name] = new_name

class _OpTypeRename(object):
  _OPTYPE_RENAME_MAP = {
    "FullyConnected": "FullyConnectedOperator",
    "Quantize": "QuantizeOperator",
    "DepthwiseConv2d": "DepthwiseSeparableConvOperator",
    "AvgPool2d": "AvgPoolOperator",
    "MinPool2d": "MinPoolOperator",
    "MaxPool2d": "MaxPoolOperator",
    "Dequantize": "DequantizeOperator",
    "Reshape": "ReshapeOperator",
    "Conv2d": "Conv2dOperator",
    "Add": "AddOperator",
    "Mul": "MulOperator",
    "Sin": "SinOperator",
    "Transpose": "TransposeOperator",
    "Const": "Constant",
    "Sub": "SubOperator",
    "Tanh": "TanhOperator",
    "Concatenation": "ConcatOperator",
    "StridedSlice": "StridedSliceOperator",
    "Div": "DivOperator",
    "Sum": "SumOperator",
    "Rsqrt": "RsqrtOperator",
  }
  
  @classmethod
  def apply(cls, ugraph):
    for op_info in ugraph.ops_info.values():
      op_info.op_type = cls._OPTYPE_RENAME_MAP.get(
        op_info.op_type,
        op_info.op_type,
      )


class _GraphRewrite(object):

  @classmethod
  def apply(cls, ugraph):
    cls._handle_fully_connected(ugraph)
    cls._handle_conv_2d(ugraph)
    cls._handle_expand_dims(ugraph)
    cls._handle_concat(ugraph)

  @classmethod
  def _handle_fully_connected(cls, ugraph):
    # 1. transpose the filter to make a right mulitiplication: fc = x @ filter + bias
    # 2. if the input is not flatten, inject a reshape op
    reshape_cnt = 0
    for op_info in ugraph.get_ops_by_type('FullyConnected'):
      filter_tensor = op_info.input_tensors[1]
      filter_op = filter_tensor.op
      np_arr = filter_op.op_attr['value'].value.np_array
      filter_op.op_attr['value'].value.np_array = np_arr.T
      filter_tensor.shape = list(np_arr.T.shape)
      filter_op.output_tensors[0].shape = list(np_arr.T.shape)

      tensor_x = op_info.input_tensors[0]
      if len(tensor_x.shape) > 2:
        new_shape = [tensor_x.shape[0], reduce(lambda a, b: a*b, tensor_x.shape[1:], 1)]
        reshape_op_name = tensor_x.name.replace(":", "_") + '_Reshape' + str(reshape_cnt)
        out_tensor = deepcopy(tensor_x, {'ugraph': ugraph})
        out_tensor.name = reshape_op_name + ":0"
        out_tensor.op_name = reshape_op_name
        out_tensor.shape = new_shape
        OperationInfo(
          name=reshape_op_name,
          op_type="Reshape",
          lib_name='tflite',
          ugraph=ugraph,
          input_tensors=[tensor_x],
          output_tensors=[out_tensor],
          op_attr={
            'new_shape': new_shape
          }
        )
        reshape_cnt += 1
        op_info.input_tensors[0] = out_tensor
    topologic_order_graph(ugraph)
  
  @classmethod
  def _handle_conv_2d(cls, ugraph):
    activation_pattern = re.compile(r'^(\d+) \(\w+\)$')
    activation_map = {
      '0': 'None',
      '1': 'ReLUOperator',
      # '2': 'TFLM::TfLiteFusedActivation::kTfLiteActRelu1',
      '3': 'ReLU6Operator',
      # '4': 'TFLM::TfLiteFusedActivation::kTfLiteActTanh',
      # '5': 'TFLM::TfLiteFusedActivation::kTfLiteActSignBit',
      # '6': 'TFLM::TfLiteFusedActivation::kTfLiteActSigmoid',
    }
    for i, op_info in enumerate(ugraph.get_ops_by_type('Conv2d')):
      act_idx = activation_pattern.match(
        op_info.op_attr['FusedActivationFunction']
      ).group(1)
      act_op_type = activation_map.get(act_idx)
      if act_op_type is None:
        raise ValueError(
          'legalization fail, unknown activation: {}'.format(
            op_info.op_attr['FusedActivationFunction']
            )
        )
      elif act_op_type == 'None':
        # no activation is set, ignore
        continue
      else:
        ori_out_tensor = op_info.output_tensors[0]
        act_op_name = '{}/{}'.format(op_info.name, act_op_type.replace('Operator', ''))
        act_tensor = TensorInfo(
          name='{}:0'.format(act_op_name),
          op_name=act_op_name,
          dtype=ori_out_tensor.dtype,
          shape=ori_out_tensor.shape[:],
          ugraph=ugraph,
          attributes=dict(ori_out_tensor.attributes),
        )
        OperationInfo(
          name=act_op_name,
          input_tensors=[ori_out_tensor],
          output_tensors=[act_tensor],
          op_type=act_op_type,
          lib_name=ugraph.lib_name,
          ugraph=ugraph,
          op_attr={}
        )
        for consumer_op in ori_out_tensor.op.output_nodes:
          for i, input_tensor in enumerate(consumer_op.input_tensors):
            if input_tensor.name == ori_out_tensor.name:
              consumer_op.input_tensors[i] = act_tensor
    topologic_order_graph(ugraph)

  @classmethod
  def _handle_expand_dims(cls, ugraph):
    """
    replace all ExpandDims operator with Reshape
    """
    ops_to_remove = set()
    for op in ugraph.get_ops_by_type("ExpandDims"):
      new_shape = op.output_tensors[0].shape
      reshape_op = OperationInfo(
        name=f'{op.name}_AsReshape',
        op_type="Reshape",
        lib_name="tflite",
        ugraph=ugraph,
        input_tensors=op.input_tensors[:1],
        output_tensors=[
          TensorInfo(
            name=f'{op.name}_AsReshape:0',
            op_name=f'{op.name}_AsReshape',
            dtype=op.output_tensors[0].dtype,
            shape=new_shape,
            ugraph=ugraph,
            attributes=deepcopy(op.output_tensors[0].attributes),
          )
        ],
        op_attr={
          'new_shape': new_shape
        }
      )
      out_tensor_name = op.output_tensors[0].name
      for out_op in op.output_nodes:
        for i, in_tensor in enumerate(out_op.input_tensors):
          if in_tensor.name == out_tensor_name:
            out_op.input_tensors[i] = reshape_op.output_tensors[0]
      ops_to_remove.add(op.name)
    if ops_to_remove:
      prune_graph(ugraph, inplace=True)

  @classmethod
  def _handle_concat(cls, ugraph):
    """
    expand concat op with more than 2 inputs to multiple concat

    ex: concat(a, b, c, axis=3) -> concat(concat(a, b, axis=3), c, axis=3)
    """
    def _expand(op):
      cnt = 1
      axis = op.op_attr["axis"]
      new_shape = op.input_tensors[0].shape[:]
      new_shape[axis] += op.input_tensors[1].shape[axis]
      op_prime = OperationInfo(
        name=f"{op.name}_{cnt:02d}",
        op_type=op.op_type,
        lib_name=op.lib_name,
        ugraph=ugraph,
        input_tensors=op.input_tensors[:2],
        output_tensors=[
          TensorInfo(
            name=f'{op.name}_{cnt:02d}:0',
            op_name=f"{op.name}_{cnt:02d}",
            dtype=op.output_tensors[0].dtype,
            shape=new_shape,
            ugraph=ugraph
          ),
        ],
        op_attr=deepcopy(op.op_attr),
      )
      for tensor in op.input_tensors[2:]:
        new_shape = op_prime.output_tensors[0].shape[:]
        new_shape[axis] += tensor.shape[axis]
        cnt += 1
        op_prime = OperationInfo(
          name=f"{op.name}_{cnt:02d}",
          op_type=op.op_type,
          lib_name=op.lib_name,
          ugraph=ugraph,
          input_tensors=[op_prime.output_tensors[0], tensor],
          output_tensors=[
            TensorInfo(
              name=f'{op.name}_{cnt:02d}:0',
              op_name=f"{op.name}_{cnt:02d}",
              dtype=op.output_tensors[0].dtype,
              shape=new_shape,
              ugraph=ugraph
            ),
          ],
          op_attr=deepcopy(op.op_attr),
        )
      for out_op in op.output_nodes:
        for i, tensor in enumerate(out_op.input_tensors):
          if tensor.name == op.output_tensors[0].name:
            out_op.input_tensors[i] = op_prime.output_tensors[0]
      prune_graph(ugraph, inplace=True)
    for op in ugraph.get_ops_by_type("Concatenation"):
      if len(op.input_tensors) > 2:
        _expand(op)