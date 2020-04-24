from __future__ import absolute_import

import os
import re

import numpy as np
from utensor_cgen.frontend import FrontendSelector
from utensor_cgen.frontend.base import Parser
from utensor_cgen.ir.base import OperationInfo, TensorInfo, uTensorGraph
from utensor_cgen.ir.converter import (AttrValueConverter,
                                       GenericTensorConverterMixin)
from utensor_cgen.utils import topologic_order_graph

from .tflite_flatbuffer.ActivationFunctionType import ActivationFunctionType
from .tflite_flatbuffer.BuiltinOperator import BuiltinOperator
from .tflite_flatbuffer.CustomOptionsFormat import CustomOptionsFormat
from .tflite_flatbuffer.FullyConnectedOptionsWeightsFormat import \
    FullyConnectedOptionsWeightsFormat
from .tflite_flatbuffer.Model import Model

tensor_np_type = dict()
tensor_np_type[0] = np.dtype("float32")
tensor_np_type[1] = np.dtype("float16")
tensor_np_type[2] = np.dtype("int32")
tensor_np_type[3] = np.dtype("uint8")
tensor_np_type[4] = np.dtype("uint64")
tensor_np_type[5] = np.dtype("str")
tensor_np_type[6] = np.dtype("bool")
tensor_np_type[7] = np.dtype("int16")
tensor_np_type[8] = np.dtype("cdouble")
tensor_np_type[9] = np.dtype("int8")


builtin_ops = {v: k for k, v in BuiltinOperator.__dict__.items()}


def class_option2str(obj, idx):
  names_lookup = {v: k for k, v in obj.__dict__.items()}
  name = names_lookup[idx]
  return str(idx) + " (" + name + ")"


customOptionFormat_lookup = {v: k for k, v in CustomOptionsFormat.__dict__.items()}


def fully_connected_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.FullyConnectedOptions import FullyConnectedOptions

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
      customOptionFormat_lookup[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()
  return option_dict


def depthwise_conv2d_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.DepthwiseConv2DOptions import DepthwiseConv2DOptions

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
      customOptionFormat_lookup[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict


def reshape_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.ReshapeOptions import ReshapeOptions

    option = ReshapeOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["NewShape"] = option.NewShapeAsNumpy()
  else:
    option_dict[
      customOptionFormat_lookup[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict


def dequantize_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.DequantizeOptions import DequantizeOptions

    option = DequantizeOptions()
    builtin_data = op.BuiltinOptions()
    if builtin_data is None:
      return option_dict
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict['builtin'] = option
  else:
    option_dict[
      customOptionFormat_lookup[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def quantize_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.QuantizeOptions import QuantizeOptions

    option = QuantizeOptions()
    builtin_data = op.BuiltinOptions()
    if builtin_data is None:
      return option_dict
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict['builtin'] = option
  else:
    option_dict[
      customOptionFormat_lookup[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def pool2d_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.Pool2DOptions import Pool2DOptions

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
      customOptionFormat_lookup[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict


def argmax_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.ArgMaxOptions import ArgMaxOptions

    option = ArgMaxOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["OutputType"] = option.OutputType()
  else:
    option_dict[
      customOptionFormat_lookup[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict


op_data_func = dict()
op_data_func["QUANTIZE"] = quantize_op_data
op_data_func["DEPTHWISE_CONV_2D"] = depthwise_conv2d_op_data
op_data_func["MAX_POOL_2D"] = pool2d_op_data
op_data_func["RESHAPE"] = reshape_op_data
op_data_func["FULLY_CONNECTED"] = fully_connected_op_data
op_data_func["DEQUANTIZE"] = dequantize_op_data
op_data_func["ARG_MAX"] = argmax_op_data


@FrontendSelector.register(target_exts=[".tflite"])
class TFLiteParser(Parser):
  def parse(self, tflite_file, output_nodes=None):
    graph_name, _ = os.path.splitext(tflite_file)
    buf = open(tflite_file, "rb").read()
    buf = bytearray(buf)
    fb_model = Model.GetRootAsModel(buf, 0)

    ugraph = uTensorGraph(
      name=graph_name, output_nodes=[], lib_name="tflite", ops_info={},
    )

    self._build_graph(fb_model, ugraph)

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

      dtype = tensor_np_type[tensor.Type()]

      attributes = dict()

      quantParam = tensor.Quantization()
      if quantParam != None:
        attributes["quantizationZeros"] = list([quantParam.ZeroPointAsNumpy()])
        attributes["quantizationScales"] = list([quantParam.ScaleAsNumpy()])

      if type(tensor.ShapeAsNumpy()) == np.ndarray:
        shape = tensor.ShapeAsNumpy().tolist()
      else:
        shape = [d for d in fb_model.Buffers(12).DataAsNumpy().shape]

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
      if type(buffer_array) == int:

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
      op_type = builtin_ops[builtinOperator_code]

      node_name = str(i) + "_" + op_type

      input_tensor_names = [
        self.tensor_names_map[input_index] for input_index in op.InputsAsNumpy()
      ]
      output_tensor_names = [
        self.tensor_names_map[output_index]
        for output_index in op.OutputsAsNumpy()
      ]

      op_attr = op_data_func[op_type](op)

      OperationInfo(
        name=node_name,
        input_tensors=input_tensor_names,
        output_tensors=output_tensor_names,
        op_type=op_type,
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
