import os
import re

import numpy as np

from utensor_cgen.frontend import FrontendSelector
from utensor_cgen.frontend.base import Parser
from utensor_cgen.ir.base import OperationInfo, TensorInfo, uTensorGraph
from utensor_cgen.ir.converter import (AttrValueConverter,
                                       GenericTensorConverterMixin)
from utensor_cgen.third_party.tflite.ActivationFunctionType import \
    ActivationFunctionType
from utensor_cgen.third_party.tflite.BuiltinOperator import BuiltinOperator
from utensor_cgen.third_party.tflite.CustomOptionsFormat import \
    CustomOptionsFormat
from utensor_cgen.third_party.tflite.FullyConnectedOptionsWeightsFormat import \
    FullyConnectedOptionsWeightsFormat
from utensor_cgen.third_party.tflite.Model import Model
from utensor_cgen.utils import topologic_order_graph

_CUSTOM_OPTION_FORMAT_MAP = {v: k for k, v in CustomOptionsFormat.__dict__.items()}

def class_option2str(obj, idx):
  names_lookup = {v: k for k, v in obj.__dict__.items()}
  name = names_lookup[idx]
  return str(idx) + " (" + name + ")"

def fully_connected_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from utensor_cgen.third_party.tflite.FullyConnectedOptions import FullyConnectedOptions

    option = FullyConnectedOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["FusedActivationFunction"] = class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
    option_dict["w_formats"] = class_option2str(
      FullyConnectedOptionsWeightsFormat, option.WeightsFormat()
    )
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()
  return option_dict

def depthwise_conv2d_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from utensor_cgen.third_party.tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions

    option = DepthwiseConv2DOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["Padding"] = option.Padding()
    option_dict["StrideW"] = option.StrideW()
    option_dict["StrideH"] = option.StrideH()
    option_dict["DepthMultiplier"] = option.DepthMultiplier()
    option_dict["FusedActivationFunction"] = class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
    option_dict["DilationWFactor"] = option.DilationWFactor()
    option_dict["DilationHFactor"] = option.DilationHFactor()

  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def reshape_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from utensor_cgen.third_party.tflite.ReshapeOptions import ReshapeOptions

    option = ReshapeOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["NewShape"] = option.NewShapeAsNumpy()
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def dequantize_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from utensor_cgen.third_party.tflite.DequantizeOptions import DequantizeOptions

    option = DequantizeOptions()
    builtin_data = op.BuiltinOptions()
    if builtin_data is None:
      return option_dict
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict['builtin'] = option
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def quantize_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from utensor_cgen.third_party.tflite.QuantizeOptions import QuantizeOptions

    option = QuantizeOptions()
    builtin_data = op.BuiltinOptions()
    if builtin_data is None:
      return option_dict
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict['builtin'] = option
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def pool2d_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from utensor_cgen.third_party.tflite.Pool2DOptions import Pool2DOptions

    option = Pool2DOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["Padding"] = option.Padding()
    option_dict["StrideW"] = option.StrideW()
    option_dict["StrideH"] = option.StrideH()
    option_dict["FilterWidth"] = option.FilterWidth()
    option_dict["FilterHeight"] = option.FilterHeight()
    option_dict["FusedActivationFunction"] = class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def argmax_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from utensor_cgen.third_party.tflite.ArgMaxOptions import ArgMaxOptions

    option = ArgMaxOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["OutputType"] = option.OutputType()
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict


_OP_DATA_FUNC_MAP = dict()
_OP_DATA_FUNC_MAP["QUANTIZE"] = quantize_op_data
_OP_DATA_FUNC_MAP["DEPTHWISE_CONV_2D"] = depthwise_conv2d_op_data
_OP_DATA_FUNC_MAP["MAX_POOL_2D"] = pool2d_op_data
_OP_DATA_FUNC_MAP["RESHAPE"] = reshape_op_data
_OP_DATA_FUNC_MAP["FULLY_CONNECTED"] = fully_connected_op_data
_OP_DATA_FUNC_MAP["DEQUANTIZE"] = dequantize_op_data
_OP_DATA_FUNC_MAP["ARG_MAX"] = argmax_op_data


@FrontendSelector.register(target_exts=[".tflite"])
class TFLiteParser(Parser):
  _TENSOR_NP_TYPE = {
    0:np.dtype("float32"),
    1: np.dtype("float16"),
    2: np.dtype("int32"),
    3: np.dtype("uint8"),
    4: np.dtype("uint64"),
    5: np.dtype("str"),
    6: np.dtype("bool"),
    7: np.dtype("int16"),
    8: np.dtype("cdouble"),
    9: np.dtype("int8"),
  }
  _BUILTIN_OPS = {v: k for k, v in BuiltinOperator.__dict__.items()}

  def parse(self, tflite_file, output_nodes=None, model_name=None):
    if output_nodes is None:
      output_nodes = []
    if model_name:
      graph_name = model_name
    else:
      graph_name, _ = os.path.splitext(
        os.path.basename(tflite_file)
      )
    with open(tflite_file, "rb") as fid:
      buf = bytearray(fid.read())
    fb_model = Model.GetRootAsModel(buf, 0)

    ugraph = uTensorGraph(
      name=graph_name,
      output_nodes=output_nodes,
      lib_name="tflite",
      ops_info={},
    )
    self._build_graph(fb_model, ugraph)
    _OpRenaming.apply(ugraph)
    return ugraph

  def _build_graph(self, fb_model, ugraph):
    self.tensor_names_map = {}  # addresseed by indexi
    self._build_tensor_map(fb_model, ugraph)

    self._build_param_ops(fb_model, ugraph)
    # find and set input nodes
    self._build_input_ops(fb_model, ugraph)
    self._build_intermediate_ops(fb_model, ugraph)
    self._set_output_ops(fb_model, ugraph)

    topologic_order_graph(ugraph)

  def _set_output_ops(self, fb_model, ugraph):
    """identfy output nodes in fb_mdel
    sets output_nodes in ugraph
    Note this method will update ugraph **inplace**
    """
    subgraph = self._get_tflm_get_subgraph(fb_model)
    subgraph_outputs_indexi = subgraph.OutputsAsNumpy()  # tensor indexi
    output_node_names = set()
    for index in subgraph_outputs_indexi:
      output_node_names.add(self.tensor_names_map[index].op_name)

    ugraph.output_nodes = list(output_node_names)

  def _build_tensor_map(self, fb_model, ugraph):
    subgraph = self._get_tflm_get_subgraph(fb_model)

    for idx in range(0, subgraph.TensorsLength()):
      tensor = subgraph.Tensors(idx)
      tensor_name = tensor.Name().decode('utf8')
      if tensor_name is "" or None:
        tensor_name = "tensor_" + str(idx)

      dtype = self._TENSOR_NP_TYPE[tensor.Type()]
      attributes = dict()
      quant_params = tensor.Quantization()
      if quant_params is not None:
        zp = quant_params.ZeroPointAsNumpy()
        if zp.dtype == np.dtype('<i8'):
          zp = zp.astype('int8')
        else:
          zp = zp.astype('uint8')
        attributes["quantization_zeros"] = zp
        attributes["quantization_scales"] = quant_params.ScaleAsNumpy()

      if isinstance(tensor.ShapeAsNumpy(), np.ndarray):
        shape = tensor.ShapeAsNumpy().tolist()
      else:
        shape = list(fb_model.Buffers(12).DataAsNumpy().shape)

      self.tensor_names_map[idx] = TensorInfo(
        name=self._format_tensor_name("", tensor_name, 0),
        op_name="",
        dtype=dtype,
        shape=shape,
        attributes=attributes,
        ugraph=ugraph,
      )

  def _build_param_ops(self, fb_model, ugraph):
    """Const tensors are identified by buffer_index == 0. These tensors are converted to Const Op and added to ugraph
    """
    subgraph = self._get_tflm_get_subgraph(fb_model)

    for idx in range(0, subgraph.TensorsLength()):
      tensor = subgraph.Tensors(idx)
      buffer_index = tensor.Buffer()

      # buffer_index == 0 if intermediate
      if buffer_index == 0:
        continue

      node_name = re.sub(r':\d+', '', self.tensor_names_map[idx].name) + "_Const"
      dtype = self.tensor_names_map[idx].dtype

      buffer_array = fb_model.Buffers(buffer_index).DataAsNumpy()
      if isinstance(buffer_array, int):
        continue  # somehow, sometimes, the buffer contains no data, likely to be an intermediate tensor
      buffer_content = fb_model.Buffers(buffer_index).DataAsNumpy().astype(dtype)

      OperationInfo(
        name=node_name,
        input_tensors=[],
        output_tensors=[self.tensor_names_map[idx]],
        op_type="Const",
        lib_name="tflm",
        ugraph=ugraph,
        op_attr={
          "value": AttrValueConverter.GenericType(
            value_name="tensor",
            value=GenericTensorConverterMixin.GenericType(
              np_array=buffer_content
            ),
          )
        },
      )

      self._set_tensor_node(idx, node_name)

  def _build_input_ops(self, fb_model, ugraph):
    """Find placeholders
    Attach placeholders to input tensors
    Note this method will update inputs **inplace**
    """
    subgraph = self._get_tflm_get_subgraph(fb_model)
    subgraph_inputs_indexi = subgraph.InputsAsNumpy()
    for index in subgraph_inputs_indexi:
      node_name = self.tensor_names_map[index].name + "_Placeholder"
      self._set_tensor_node(index, node_name)
      OperationInfo(
        name=node_name,
        input_tensors=[],
        output_tensors=[self.tensor_names_map[index]],
        op_type="Placeholder",
        ugraph=ugraph,
        lib_name="tflm",
        op_attr={},
      )

  def _build_intermediate_ops(self, fb_model, ugraph):
    """Build all intermediate nodes
    """
    subgraphs_len = fb_model.SubgraphsLength()
    assert subgraphs_len == 1, "only 1 subgraph is supported"
    subgraph = fb_model.Subgraphs(0)
    for i in range(0, subgraph.OperatorsLength()):
      # topological order, op-index defined by schema
      # BuiltinOperator: https://github.com/tensorflow/tensorflow/blob/031804922d8f4d18b61e3ad077f9f1b69273ff21/tensorflow/lite/schema/schema_v3.fbs#L71
      op = subgraph.Operators(i)
      local_op_code = op.OpcodeIndex()
      global_op_code = fb_model.OperatorCodes(local_op_code)
      builtinOperator_code = global_op_code.BuiltinCode()
      op_type = self._BUILTIN_OPS[builtinOperator_code]

      node_name = str(i) + "_" + op_type

      input_tensor_names = [
        self.tensor_names_map[input_index] for input_index in op.InputsAsNumpy()
      ]
      output_tensor_names = [
        self.tensor_names_map[output_index]
        for output_index in op.OutputsAsNumpy()
      ]

      op_attr = _OP_DATA_FUNC_MAP[op_type](op)

      OperationInfo(
        name=node_name,
        input_tensors=input_tensor_names,
        output_tensors=output_tensor_names,
        op_type=self._format_op_type(op_type),
        ugraph=ugraph,
        lib_name="tflm",
        op_attr=op_attr,
      )

      for tensor_index in op.OutputsAsNumpy():
        self._set_tensor_node(tensor_index, node_name)

  def _get_tflm_get_subgraph(self, fb_model):
    subgraphs_len = fb_model.SubgraphsLength()
    assert subgraphs_len == 1, "only 1 subgraph is supported"
    subgraph = fb_model.Subgraphs(0)

    return subgraph

  def _set_tensor_node(self, idx, name):
    assert self.tensor_names_map[idx].op_name == ""
    self.tensor_names_map[idx].op_name = name

  def _format_node_name(self, node_name, op_type, op_cnt):
    if node_name == "":
      node_name = "{}_{}".format(op_type, op_cnt)
    return re.sub(r"[\.:/]", "_", node_name)

  def _format_tensor_name(self, name, node_name, offset):
    if re.match(r"[a-zA-Z][a-zA-Z0-9]*:[0-9]+", name):
      return name
    return "{}:{}".format(node_name, offset)

  def _format_op_type(self, op_type):
    return ''.join(map(lambda s: s.capitalize(), op_type.split('_')))


class _OpRenaming(object):
  _OP_NAMES_MAP = {
    "Quantize": "QuantizeOperator",
    "DepthwiseConv2d": "DepthwiseSeparableConvOperator",
    "MaxPool2d": "MaxPoolOperator",
    "Dequantize": "DequantizeOperator",
    "FullyConnected": "FullyConnectedOperator"
  }

  @classmethod
  def apply(cls, ugraph):
    for op_info in ugraph.ops_info.values():
      op_info.op_type = cls._OP_NAMES_MAP.get(
        op_info.op_type, op_info.op_type
      )
